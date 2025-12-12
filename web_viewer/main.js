import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

/* =========================
   BASIC SETUP
   ========================= */

console.log('main.js running');

const container = document.getElementById('container');

// Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
container.appendChild(renderer.domElement);

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

// Camera
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  100000
);
camera.position.set(0, -5, 2);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Resize handling
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.6));

/* =========================
   RAYCASTING
   ========================= */

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const cameraMarkers = [];

/* =========================
   IMAGE OVERLAY
   ========================= */

const overlay = document.getElementById('imageOverlay');
const overlayImg = document.getElementById('cameraImage');
const overlayLabel = document.getElementById('imageLabel');
const overlayWarning = document.getElementById('imageWarning');

document.getElementById('closeImage').onclick = () => {
  overlay.classList.add('hidden');
};

/* =========================
   LOAD POINT CLOUD
   ========================= */

const plyLoader = new PLYLoader();

plyLoader.load(
  'point_cloud.ply',
  geometry => {
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;

    console.log('PLY bounding box:', bbox.min, bbox.max);

    const material = new THREE.PointsMaterial({
      size: 0.02,
      vertexColors: true,
      sizeAttenuation: true
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    bbox.getCenter(controls.target);
    controls.update();
  },
  undefined,
  err => {
    console.error('Failed to load PLY:', err);
  }
);

/* =========================
   LOAD CAMERAS
   ========================= */

fetch('cameras.json')
  .then(r => r.json())
  .then(json => initCameras(json))
  .catch(err => console.error('Failed to load cameras.json:', err));

function initCameras(json) {
  const cams = json.document.chunk.cameras.camera;
  console.log('Cameras loaded:', cams.length);

  cams.forEach((cam, i) => {
    if (!cam.transform) return;

    const nums = cam.transform.split(' ').map(Number);
    if (nums.length !== 16) {
      console.warn('Invalid transform for', cam['@label']);
      return;
    }

    /*
      Metashape:
      - column-major
      - world-to-camera
      → invert to get camera-to-world
    */
    const m = new THREE.Matrix4();
    m.set(
      nums[0], nums[4], nums[8],  nums[12],
      nums[1], nums[5], nums[9],  nums[13],
      nums[2], nums[6], nums[10], nums[14],
      nums[3], nums[7], nums[11], nums[15]
    );

    const camToWorld = m.clone().invert();

    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    camToWorld.decompose(pos, quat, scale);

    if (i < 3) {
      console.log(`Camera ${cam['@label']} position:`, pos);
    }

    // Camera frustum marker
    const marker = new THREE.Mesh(
      new THREE.ConeGeometry(0.05, 0.15, 8),
      new THREE.MeshBasicMaterial({ color: 0xffaa00 })
    );

    marker.position.copy(pos);
    marker.quaternion.copy(quat);
    marker.rotateX(Math.PI / 2);

    marker.userData = { cam, camToWorld };
    scene.add(marker);
    cameraMarkers.push(marker);
  });
}

/* =========================
   PICKING
   ========================= */

renderer.domElement.addEventListener('click', event => {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(cameraMarkers);
  if (hits.length > 0) {
    selectCamera(hits[0].object);
  }
});

/* =========================
   CAMERA SELECTION
   ========================= */

function selectCamera(marker) {
  const { cam, camToWorld } = marker.userData;

  // Move viewer camera
  camera.matrixAutoUpdate = false;
  camera.matrix.copy(camToWorld);
  camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
  camera.matrixAutoUpdate = true;

  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).add(forward);
  controls.update();

  // Image overlay
  overlay.classList.remove('hidden');
  overlayLabel.textContent = cam['@label'];
  overlayWarning.classList.add('hidden');

  const imgPath = `../data/${cam['@label']}.jpg`;
  overlayImg.src = imgPath;

  // Apply EXIF orientation
  if (String(cam.orientation) === '6') {
    overlayImg.style.transform = 'rotate(90deg)';
  } else {
    overlayImg.style.transform = 'none';
    if (cam.orientation) {
      console.warn(`Unhandled orientation ${cam.orientation} for ${cam['@label']}`);
    }
  }

  overlayImg.onerror = () => {
    overlayWarning.textContent = `Missing image: ${imgPath}`;
    overlayWarning.classList.remove('hidden');
  };
}

/* =========================
   RENDER LOOP
   ========================= */

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

animate();
