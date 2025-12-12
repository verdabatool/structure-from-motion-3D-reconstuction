import os
import glob
import numpy as np
import cv2

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# --------- PATHS / CONFIG ---------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PLY_PATH = os.path.join(OUTPUT_DIR, "multiview_point_cloud.ply")
POSES_PATH = os.path.join(OUTPUT_DIR, "multiview_camera_poses.npy")

LOWE_RATIO = 0.75
ESSENTIAL_RANSAC_THRESH = 1.0
PNP_REPROJ_ERROR = 4.0        # px
MIN_PNP_INLIERS = 40          # minimum inliers to accept a camera
MIN_TRIANGULATION_ANGLE = np.deg2rad(1.0)  # radians
REPROJ_ERROR_TRIANG = 3.0     # px
BA_OUTLIER_THRESH = 10.0      # px, drop obs with larger initial error from BA
# ----------------------------------


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_intrinsics_from_exif(image_path):
    """
    Compute camera intrinsics K using EXIF metadata:

    Requires:
        - FocalLength (mm)
        - FocalLengthIn35mmFilm
        - ExifImageWidth, ExifImageHeight

    Uses 36mm full-frame width and 35mm equivalence to estimate sensor size.
    """
    try:
        from PIL import Image, ExifTags
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required for EXIF-based intrinsics. "
            "Install with `pip install pillow`."
        ) from e

    img = Image.open(image_path)
    exif = img._getexif()
    if exif is None:
        raise RuntimeError("No EXIF data found in image; cannot compute intrinsics.")

    focal_length_mm = None
    focal_length_35mm = None
    width_px = None
    height_px = None

    for tag, val in exif.items():
        key = ExifTags.TAGS.get(tag, tag)
        if key == "FocalLength":
            if isinstance(val, tuple):
                focal_length_mm = float(val[0]) / float(val[1])
            else:
                focal_length_mm = float(val)
        elif key == "FocalLengthIn35mmFilm":
            focal_length_35mm = float(val)
        elif key == "ExifImageWidth":
            width_px = int(val)
        elif key == "ExifImageHeight":
            height_px = int(val)

    if focal_length_mm is None or focal_length_35mm is None:
        raise RuntimeError(
            "Missing FocalLength or FocalLengthIn35mmFilm in EXIF; "
            "cannot compute physical intrinsics."
        )
    if width_px is None or height_px is None:
        raise RuntimeError(
            "Missing ExifImageWidth/ExifImageHeight in EXIF; "
            "cannot compute intrinsics."
        )

    # Compute sensor size from 35mm equivalence
    sensor_width_mm = 36.0 * (focal_length_mm / focal_length_35mm)
    sensor_height_mm = sensor_width_mm * (height_px / width_px)

    fx = focal_length_mm / sensor_width_mm * width_px
    fy = focal_length_mm / sensor_height_mm * height_px
    cx = width_px / 2.0
    cy = height_px / 2.0

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0, 1]
    ], dtype=np.float64)

    print("[K] EXIF intrinsics from", os.path.basename(image_path))
    print(K)
    return K


# def load_all_images():
#     pattern = os.path.join(DATA_DIR, "*.JPG")
#     files = sorted(glob.glob(pattern))
#     if len(files) < 2:
#         raise RuntimeError(f"Need at least 2 images in {DATA_DIR}, found {len(files)}")
#
#     images_color = []
#     images_gray = []
#     for f in files:
#         img = cv2.imread(f, cv2.IMREAD_COLOR)
#         if img is None:
#             raise RuntimeError(f"Failed to read image: {f}")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         images_color.append(img)
#         images_gray.append(gray)
#
#     print(f"[LOAD] Loaded {len(images_color)} images from Data/")
#     return images_color, images_gray, files


MAX_DIM = 1200  # or 800

def load_all_images(K):
    pattern = os.path.join(DATA_DIR, "*.jpg")
    files = sorted(glob.glob(pattern))

    images_color = []
    images_gray = []

    # Get EXIF original size (needed for scaling)
    original_img = cv2.imread(files[0])
    orig_h, orig_w = original_img.shape[:2]

    # Compute scale factor based on original EXIF size
    scale = MAX_DIM / max(orig_h, orig_w)

    # ---- SCALE THE INTRINSICS HERE ----
    if scale < 1.0:
        K = K.copy()
        K[0, 0] *= scale   # fx
        K[1, 1] *= scale   # fy
        K[0, 2] *= scale   # cx
        K[1, 2] *= scale   # cy

    # Now actually resize images
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images_color.append(img)
        images_gray.append(gray)

    print(f"[LOAD] Loaded {len(images_color)} images, resized by scale={scale:.3f}")
    return images_color, images_gray, files, K



def create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    else:
        return cv2.xfeatures2d.SIFT_create()



def extract_features(images_gray):
    sift = create_sift()
    keypoints_list = []
    descriptors_list = []
    for i, img in enumerate(images_gray):
        kps, desc = sift.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            raise RuntimeError(f"No SIFT features found in image index {i}")
        keypoints_list.append(kps)
        descriptors_list.append(desc.astype(np.float32))
        print(f"[SIFT] Image {i}: {len(kps)} keypoints")
    return keypoints_list, descriptors_list


def create_flann():
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def match_descriptors(desc1, desc2, ratio=LOWE_RATIO):
    flann = create_flann()
    knn = flann.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_essential(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(
        pts1, pts2, cameraMatrix=K,
        method=cv2.RANSAC, prob=0.999,
        threshold=ESSENTIAL_RANSAC_THRESH
    )
    if E is None:
        raise RuntimeError("cv2.findEssentialMat failed.")
    mask = mask.ravel().astype(bool)
    print(f"[E] Inliers: {np.count_nonzero(mask)}/{len(pts1)}")
    return E, mask


def decompose_essential_to_poses(E):
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1

    poses = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t),
    ]
    return poses


def cheirality_count(R, t, X):
    """
    X in world frame = camera 0 frame.
    Count points with positive depth in cam0 and cam1.
    """
    z0 = X[:, 2]
    X1 = (R @ X.T + t.reshape(3, 1)).T
    z1 = X1[:, 2]
    mask = (z0 > 0) & (z1 > 0)
    return np.count_nonzero(mask), mask


def choose_initial_pose(E, pts0, pts1, K):
    poses = decompose_essential_to_poses(E)

    pts0_n = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts1_n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    best_count = -1
    best = None

    for idx, (R, t) in enumerate(poses):
        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = np.hstack([R, t.reshape(3, 1)])
        X_h = cv2.triangulatePoints(P0, P1, pts0_n.T, pts1_n.T)
        X = (X_h[:3, :] / X_h[3, :]).T

        # Safety: drop NaNs / Infs here too
        valid = np.isfinite(X).all(axis=1)
        if not np.all(valid):
            X = X[valid]
        if X.shape[0] == 0:
            count = 0
            mask = np.zeros(pts0_n.shape[0], dtype=bool)
        else:
            count, mask = cheirality_count(R, t, X)

        print(f"[INIT] Pose {idx}: {count} points in front")
        if count > best_count:
            best_count = count
            best = (R, t, X, mask)

    R_best, t_best, X, mask = best
    print(f"[INIT] Selected pose with {best_count} points in front")
    return R_best, t_best, X, mask


def reprojection_errors(X, R, t, K, pts):
    """
    X: Nx3 in world frame
    pts: Nx2 pixels
    Returns: errors (N,)
    """
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    X_cam = (R @ X.T + t.reshape(3, 1)).T
    # Avoid divide-by-zero in depth
    z = X_cam[:, 2]
    valid_z = np.abs(z) > 1e-9
    X_cam = X_cam.copy()
    X_cam[~valid_z, 2] = 1e-9

    x = X_cam[:, 0] / X_cam[:, 2]
    y = X_cam[:, 1] / X_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x + cx
    v = fy * y + cy

    proj = np.stack([u, v], axis=1)
    if proj.shape[0] != pts.shape[0]:
        raise RuntimeError(
            f"Reprojection size mismatch: proj {proj.shape}, pts {pts.shape}"
        )

    err = np.sqrt(np.sum((pts - proj) ** 2, axis=1))
    return err


def triangulate_between_cams(R0, t0, R1, t1, pts0, pts1, K):
    """
    Triangulate N points seen by two cameras (R0,t0),(R1,t1).
    pts0, pts1: Nx2 pixels.
    Returns: X: Nx3 world coordinates.
    """
    if pts0.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts0_n = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts1_n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    P0 = np.hstack([R0, t0.reshape(3, 1)])
    P1 = np.hstack([R1, t1.reshape(3, 1)])

    X_h = cv2.triangulatePoints(P0, P1, pts0_n.T, pts1_n.T)
    X = (X_h[:3, :] / X_h[3, :]).T
    return X


def triangulation_angles(R0, t0, R1, t1, X):
    """
    Angle between viewing rays for each point.
    X: Nx3 world coordinates.
    Returns: angles (N,) in radians.
    """
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    C0 = -R0.T @ t0
    C1 = -R1.T @ t1

    v0 = X - C0.reshape(1, 3)
    v1 = X - C1.reshape(1, 3)

    v0 /= np.linalg.norm(v0, axis=1, keepdims=True) + 1e-9
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9

    cosang = np.clip(np.sum(v0 * v1, axis=1), -1.0, 1.0)
    return np.arccos(cosang)


def save_ply(points_3d, colors_bgr, filepath):
    ensure_output_dir()
    pts = np.asarray(points_3d, dtype=np.float32)
    cols = np.asarray(colors_bgr, dtype=np.uint8)
    assert pts.shape[0] == cols.shape[0]

    n = pts.shape[0]
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            r, g, b = int(c[2]), int(c[1]), int(c[0])
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")
    print(f"[OUTPUT] Saved point cloud: {filepath}")


# ----------------- BUNDLE ADJUSTMENT (FULL BA) -----------------

def build_ba_problem(points3D_arr, R_list, t_list,
                     kp_to_3d, keypoints_list, K):
    """
    Build the BA data structures:
    - Cameras: fixed vs optimized
    - BA points (subset of points3D with >=2 observations)
    - Observations (camera index, point index, measured pixel)
    """

    from collections import defaultdict

    # 1) Determine valid cameras (those that have a pose)
    n_images = len(R_list)
    valid_cams = [i for i in range(n_images)
                  if R_list[i] is not None and t_list[i] is not None]
    if len(valid_cams) < 2:
        print("[BA] Not enough cameras with valid poses. Skipping BA.")
        return None

    # Fix baseline cameras: 0 and 1 if they exist & valid
    fixed_cams = set()
    if 0 in valid_cams:
        fixed_cams.add(0)
    if 1 in valid_cams:
        fixed_cams.add(1)

    opt_cams = [i for i in valid_cams if i not in fixed_cams]

    print(f"[BA] Valid cameras: {valid_cams}")
    print(f"[BA] Fixed cameras: {sorted(list(fixed_cams))}")
    print(f"[BA] Optimized cameras: {opt_cams}")

    if len(opt_cams) == 0:
        print("[BA] No cameras to optimize (only fixed ones). Skipping BA.")
        return None

    # 2) Build observation lists per point
    point_obs = defaultdict(list)  # p_idx -> list of (img_idx, kp_idx)
    for (img_idx, kp_idx), p_idx in kp_to_3d.items():
        if p_idx < 0 or p_idx >= points3D_arr.shape[0]:
            continue
        point_obs[p_idx].append((img_idx, kp_idx))

    # 3) Build BA point set: we only keep points with >=2 observations
    ba_points = []
    ba_index_to_old = []
    # For building observations
    obs_cam_idx = []
    obs_pt_idx = []
    obs_uv = []

    # We'll use only cameras that are valid (fixed or optimized)
    cam_is_used = set(valid_cams)

    for old_idx, obs_list in point_obs.items():
        # Require at least two observations
        if len(obs_list) < 2:
            continue

        X = points3D_arr[old_idx]
        if not np.all(np.isfinite(X)):
            continue

        # Check that at least one observation is from a valid camera
        obs_valid = [(img_idx, kp_idx) for (img_idx, kp_idx) in obs_list
                     if img_idx in cam_is_used]
        if len(obs_valid) < 2:
            continue

        new_idx = len(ba_points)
        ba_points.append(X)
        ba_index_to_old.append(old_idx)

        for (img_idx, kp_idx) in obs_valid:
            kp = keypoints_list[img_idx][kp_idx]
            u, v = kp.pt
            obs_cam_idx.append(img_idx)
            obs_pt_idx.append(new_idx)
            obs_uv.append([u, v])

    ba_points = np.array(ba_points, dtype=np.float64)
    obs_cam_idx = np.array(obs_cam_idx, dtype=np.int32)
    obs_pt_idx = np.array(obs_pt_idx, dtype=np.int32)
    obs_uv = np.array(obs_uv, dtype=np.float64)

    n_points = ba_points.shape[0]
    n_obs = obs_uv.shape[0]

    print(f"[BA] BA points: {n_points}")
    print(f"[BA] BA observations: {n_obs}")

    if n_points == 0 or n_obs == 0:
        print("[BA] No points/observations to optimize. Skipping BA.")
        return None

    # 4) Map cameras to optimization indices
    cam_opt_index = {}  # img_idx -> index in [0..n_opt_cams-1] or None if fixed
    for idx, img_idx in enumerate(opt_cams):
        cam_opt_index[img_idx] = idx
    for img_idx in fixed_cams:
        cam_opt_index[img_idx] = None  # not optimized

    # 5) Build initial camera parameter vector (Rodrigues + t) for optimized cams
    n_opt_cams = len(opt_cams)
    cam_params = np.zeros((n_opt_cams, 6), dtype=np.float64)
    for k, img_idx in enumerate(opt_cams):
        R = R_list[img_idx]
        t = t_list[img_idx]
        rvec, _ = cv2.Rodrigues(R)
        cam_params[k, :3] = rvec.ravel()
        cam_params[k, 3:] = t.ravel()

    # 6) Fixed cameras: keep R, t as is
    fixed_R = {}
    fixed_t = {}
    for img_idx in fixed_cams:
        fixed_R[img_idx] = R_list[img_idx]
        fixed_t[img_idx] = t_list[img_idx]

    # 7) Remove gross outlier observations before BA using current structure
    #    This prevents BA from blowing up.
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def project_point(R, t, X):
        X_cam = R @ X + t
        if abs(X_cam[2]) < 1e-9:
            Z = 1e-9
        else:
            Z = X_cam[2]
        u = fx * (X_cam[0] / Z) + cx
        v = fy * (X_cam[1] / Z) + cy
        return np.array([u, v])

    # Compute initial reprojection error per observation
    all_errors = []
    for o in range(n_obs):
        img_idx = obs_cam_idx[o]
        pt_ba_idx = obs_pt_idx[o]
        X = ba_points[pt_ba_idx]

        if img_idx in fixed_cams:
            R = fixed_R[img_idx]
            t = fixed_t[img_idx]
        else:
            k = cam_opt_index[img_idx]
            rvec = cam_params[k, :3]
            tvec = cam_params[k, 3:]
            R, _ = cv2.Rodrigues(rvec)
            t = tvec

        proj = project_point(R, t, X)
        err = np.linalg.norm(obs_uv[o] - proj)
        all_errors.append(err)

    all_errors = np.array(all_errors)
    keep_obs = all_errors < BA_OUTLIER_THRESH
    n_pruned = np.count_nonzero(~keep_obs)
    if n_pruned > 0:
        print(f"[BA] Removing {n_pruned} gross outlier observations (> {BA_OUTLIER_THRESH} px)")
        obs_cam_idx = obs_cam_idx[keep_obs]
        obs_pt_idx = obs_pt_idx[keep_obs]
        obs_uv = obs_uv[keep_obs]
        all_errors = all_errors[keep_obs]
        n_obs = obs_uv.shape[0]

    # 8) Build sparse Jacobian pattern
    n_cam_params = 6 * n_opt_cams
    n_point_params = 3 * n_points
    n_params = n_cam_params + n_point_params
    n_residuals = 2 * n_obs

    J = lil_matrix((n_residuals, n_params), dtype=int)

    for o in range(n_obs):
        row0 = 2 * o
        row1 = row0 + 1
        img_idx = obs_cam_idx[o]
        pt_idx = obs_pt_idx[o]

        # Camera block
        cam_k = cam_opt_index[img_idx]
        if cam_k is not None:
            base_c = 6 * cam_k
            for c in range(6):
                J[row0, base_c + c] = 1
                J[row1, base_c + c] = 1

        # Point block
        base_p = n_cam_params + 3 * pt_idx
        for k in range(3):
            J[row0, base_p + k] = 1
            J[row1, base_p + k] = 1

    J = J.tocsr()

    # 9) Build initial parameter vector
    x0 = np.hstack([
        cam_params.ravel(),   # optimized camera parameters
        ba_points.ravel()     # all BA points
    ])

    ba_data = {
        "K": K,
        "opt_cams": opt_cams,
        "fixed_cams": fixed_cams,
        "cam_opt_index": cam_opt_index,
        "fixed_R": fixed_R,
        "fixed_t": fixed_t,
        "obs_cam_idx": obs_cam_idx,
        "obs_pt_idx": obs_pt_idx,
        "obs_uv": obs_uv,
        "n_opt_cams": n_opt_cams,
        "n_points": n_points,
        "n_params": n_params,
        "n_residuals": n_residuals,
        "J_sparsity": J,
        "x0": x0,
        "ba_index_to_old": ba_index_to_old,
    }

    return ba_data


def ba_residuals(x, ba_data):
    """
    Compute reprojection residuals vector for BA.
    """
    K = ba_data["K"]
    opt_cams = ba_data["opt_cams"]
    fixed_cams = ba_data["fixed_cams"]
    cam_opt_index = ba_data["cam_opt_index"]
    fixed_R = ba_data["fixed_R"]
    fixed_t = ba_data["fixed_t"]
    obs_cam_idx = ba_data["obs_cam_idx"]
    obs_pt_idx = ba_data["obs_pt_idx"]
    obs_uv = ba_data["obs_uv"]
    n_opt_cams = ba_data["n_opt_cams"]
    n_points = ba_data["n_points"]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Unpack parameters
    cam_params = x[: 6 * n_opt_cams].reshape((n_opt_cams, 6))
    pts_params = x[6 * n_opt_cams :].reshape((n_points, 3))

    # Build camera matrices
    R_cams = {}
    t_cams = {}

    # Optimized cameras
    for k, img_idx in enumerate(opt_cams):
        rvec = cam_params[k, :3]
        tvec = cam_params[k, 3:]
        R, _ = cv2.Rodrigues(rvec)
        R_cams[img_idx] = R
        t_cams[img_idx] = tvec

    # Fixed cameras
    for img_idx in fixed_cams:
        R_cams[img_idx] = fixed_R[img_idx]
        t_cams[img_idx] = fixed_t[img_idx]

    # Compute residuals
    n_obs = obs_uv.shape[0]
    residuals = np.zeros(2 * n_obs, dtype=np.float64)

    for o in range(n_obs):
        img_idx = obs_cam_idx[o]
        pt_idx = obs_pt_idx[o]
        X = pts_params[pt_idx]

        R = R_cams[img_idx]
        t = t_cams[img_idx]

        X_cam = R @ X + t
        Z = X_cam[2] if abs(X_cam[2]) > 1e-9 else 1e-9
        u = fx * (X_cam[0] / Z) + cx
        v = fy * (X_cam[1] / Z) + cy

        u_obs, v_obs = obs_uv[o]
        residuals[2 * o] = u_obs - u
        residuals[2 * o + 1] = v_obs - v

    return residuals


def run_bundle_adjustment(points3D_arr, R_list, t_list,
                          kp_to_3d, keypoints_list, K):
    """
    Perform full BA over all camera poses (except fixed) and 3D points.
    Returns updated (points3D_arr, R_list, t_list).
    """
    ba_data = build_ba_problem(points3D_arr, R_list, t_list,
                               kp_to_3d, keypoints_list, K)
    if ba_data is None:
        print("[BA] Skipping BA.")
        return points3D_arr, R_list, t_list

    x0 = ba_data["x0"]
    J = ba_data["J_sparsity"]

    print("[BA] Starting SciPy least_squares BA...")
    res = least_squares(
        fun=ba_residuals,
        x0=x0,
        args=(ba_data,),
        jac_sparsity=J,
        method="trf",
        verbose=2,
        max_nfev=100,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    print("[BA] Done. Success:", res.success, "Status:", res.status)
    print("[BA] Final cost:", res.cost, " RMS per obs:", np.sqrt(res.cost / (ba_data["n_residuals"] / 2)))

    # Unpack optimized parameters
    n_opt_cams = ba_data["n_opt_cams"]
    n_points = ba_data["n_points"]
    opt_cams = ba_data["opt_cams"]
    ba_index_to_old = ba_data["ba_index_to_old"]

    x_opt = res.x
    cam_params_opt = x_opt[: 6 * n_opt_cams].reshape((n_opt_cams, 6))
    pts_params_opt = x_opt[6 * n_opt_cams :].reshape((n_points, 3))

    # Update camera poses for optimized cameras
    for k, img_idx in enumerate(opt_cams):
        rvec = cam_params_opt[k, :3]
        tvec = cam_params_opt[k, 3:]
        R_opt, _ = cv2.Rodrigues(rvec)
        R_list[img_idx] = R_opt
        t_list[img_idx] = tvec

    # Update 3D points
    for ba_idx, old_idx in enumerate(ba_index_to_old):
        points3D_arr[old_idx] = pts_params_opt[ba_idx]

    print("[BA] Updated camera poses and 3D points from BA.")
    return points3D_arr, R_list, t_list


# ----------------- MAIN PIPELINE -----------------


def main():
    ensure_output_dir()

    # ---- 1. Find files BEFORE computing K ----
    pattern = os.path.join(DATA_DIR, "*.jpg")
    files = sorted(glob.glob(pattern))
    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 images in {DATA_DIR}")
    print("Loading Images ....")
    # ---- 2. EXIF intrinsics from first FULL-RES image ----
    K = compute_intrinsics_from_exif(files[0])

    # ---- 3. Load + resize images AND scale intrinsics ----
    images_color, images_gray, files, K = load_all_images(K)

    N = len(images_color)
    h0, w0 = images_color[0].shape[:2]

    # ---- 4. Extract features on resized images ----
    keypoints_list, descriptors_list = extract_features(images_gray)

    # ---- Prepare matcher cache ----
    matches_cache = {}

    def get_matches(i, j):
        if (i, j) in matches_cache:
            return matches_cache[(i, j)]
        print(f"[MATCH] {i} -> {j}")
        m = match_descriptors(descriptors_list[i], descriptors_list[j])
        print(f"   {len(m)} good matches")
        matches_cache[(i, j)] = m
        return m

    # Global state
    points3D = []
    colors3D = []
    kp_to_3d = {}         # (img_idx, kp_idx) -> point index

    R_list = [None] * N   # world-to-camera rotation
    t_list = [None] * N   # world-to-camera translation

    # 4. Initialize with images 0 and 1
    print("\n[INIT] Two-view initialization with images 0 and 1")
    img0 = images_color[0]
    img1 = images_color[1]

    matches_01 = get_matches(0, 1)
    if len(matches_01) < 8:
        raise RuntimeError("Not enough matches between images 0 and 1")

    pts0 = np.float32([keypoints_list[0][m.queryIdx].pt for m in matches_01])
    pts1 = np.float32([keypoints_list[1][m.trainIdx].pt for m in matches_01])

    E, mask_inliers = estimate_essential(pts0, pts1, K)
    pts0_in = pts0[mask_inliers]
    pts1_in = pts1[mask_inliers]
    matches_in = [m for m, keep in zip(matches_01, mask_inliers) if keep]

    # ---- FIXED INITIALIZATION BLOCK ----

    R1, t1, X_init_raw, mask_front = choose_initial_pose(E, pts0_in, pts1_in, K)

    # Apply cheirality mask to ALL arrays, including X_init_raw
    X_init = X_init_raw[mask_front]
    pts0_front = pts0_in[mask_front]
    pts1_front = pts1_in[mask_front]
    matches_front = [m for m, keep in zip(matches_in, mask_front) if keep]

    print(f"[INIT] After cheirality: X={X_init.shape}, pts={pts0_front.shape}, matches={len(matches_front)}")

    # ---- SAFETY: Validate shapes BEFORE NaN filtering ----
    assert X_init.shape[0] == pts0_front.shape[0] == len(matches_front), \
        f"Shape mismatch immediately after cheirality:\n" \
        f"X={X_init.shape}, pts0={pts0_front.shape}, matches={len(matches_front)}"

    # ---- Remove NaN / Inf 3D points ----
    valid_init = np.isfinite(X_init).all(axis=1)
    if not np.all(valid_init):
        removed = np.size(valid_init) - np.count_nonzero(valid_init)
        print(f"[WARN] Removing {removed} invalid 3D points (NaN/Inf)")

    X_init = X_init[valid_init]
    pts0_front = pts0_front[valid_init]
    pts1_front = pts1_front[valid_init]
    matches_front = [m for m, keep in zip(matches_front, valid_init) if keep]

    print(f"[INIT] After NaN cleanup: {X_init.shape[0]} points")

    # ---- Reprojection error filtering ----
    R0 = np.eye(3)
    t0 = np.zeros(3)

    err0 = reprojection_errors(X_init, R0, t0, K, pts0_front)
    err1 = reprojection_errors(X_init, R1, t1, K, pts1_front)

    good_mask = (err0 < REPROJ_ERROR_TRIANG) & (err1 < REPROJ_ERROR_TRIANG)

    X_init = X_init[good_mask]
    pts0_front = pts0_front[good_mask]
    pts1_front = pts1_front[good_mask]
    matches_front = [m for m, keep in zip(matches_front, good_mask) if keep]

    print(f"[INIT] After reprojection filtering: {X_init.shape[0]} points")

    # ---- Final NaN/Inf safety ----
    valid_init2 = np.isfinite(X_init).all(axis=1)
    if not np.all(valid_init2):
        removed2 = len(valid_init2) - np.count_nonzero(valid_init2)
        print(f"[WARN] Removing {removed2} invalid points post-reproj")

    X_init = X_init[valid_init2]
    pts0_front = pts0_front[valid_init2]
    pts1_front = pts1_front[valid_init2]
    matches_front = [m for m, keep in zip(matches_front, valid_init2) if keep]

    print(f"[INIT] Final initial 3D points: {X_init.shape[0]}")

    # ---- Store initial camera poses ----
    R_list[0] = R0
    t_list[0] = t0
    R_list[1] = R1
    t_list[1] = t1

    # ---- Build initial map ----
    for X, m in zip(X_init, matches_front):
        p_idx = len(points3D)
        points3D.append(X)

        u, v = keypoints_list[0][m.queryIdx].pt
        u_i = int(round(u))
        v_i = int(round(v))
        u_i = np.clip(u_i, 0, w0 - 1)
        v_i = np.clip(v_i, 0, h0 - 1)
        color = img0[v_i, u_i]
        colors3D.append(color)

        kp_to_3d[(0, m.queryIdx)] = p_idx
        kp_to_3d[(1, m.trainIdx)] = p_idx

    print(f"[STATE] Initial 3D points: {len(points3D)}")

    # 5. Incremental registration
    for i in range(2, N):
        print(f"\n[REGISTER] Image {i}")

        prev = i - 1

        # PnP from 2D–3D matches with previous image
        matches_prev_i = get_matches(prev, i)
        pts3d_pnp = []
        pts2d_pnp = []

        for m in matches_prev_i:
            key_prev = (prev, m.queryIdx)
            if key_prev in kp_to_3d:
                p_idx = kp_to_3d[key_prev]
                pts3d_pnp.append(points3D[p_idx])
                pts2d_pnp.append(keypoints_list[i][m.trainIdx].pt)

        pts3d_pnp = np.array(pts3d_pnp, dtype=np.float32)
        pts2d_pnp = np.array(pts2d_pnp, dtype=np.float32)

        print(f"  PnP: {len(pts3d_pnp)} 2D-3D correspondences from image {prev}")

        if len(pts3d_pnp) < MIN_PNP_INLIERS:
            print("  Not enough correspondences for robust PnP, skipping this image.")
            continue

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_pnp, pts2d_pnp, K, None,
            reprojectionError=PNP_REPROJ_ERROR,
            iterationsCount=1000,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            print("  PnP failed or not enough inliers, skipping this image.")
            continue

        inliers = inliers.ravel()
        print(f"  PnP inliers: {len(inliers)}/{len(pts3d_pnp)}")

        R_i, _ = cv2.Rodrigues(rvec)
        t_i = tvec.reshape(3)
        R_list[i] = R_i
        t_list[i] = t_i

        # After camera pose is known, triangulate NEW points
        new_points = 0

        for j in range(i):
            if R_list[j] is None:
                continue

            matches_ji = get_matches(j, i)
            R_j, t_j = R_list[j], t_list[j]
            img_j = images_color[j]
            hj, wj = img_j.shape[:2]

            pts_j = []
            pts_i = []
            match_indices = []

            for m in matches_ji:
                key_j = (j, m.queryIdx)
                key_i = (i, m.trainIdx)
                # Only triangulate points not already in the map
                if key_j in kp_to_3d or key_i in kp_to_3d:
                    continue
                pts_j.append(keypoints_list[j][m.queryIdx].pt)
                pts_i.append(keypoints_list[i][m.trainIdx].pt)
                match_indices.append(m)

            if len(pts_j) == 0:
                continue

            pts_j = np.array(pts_j, dtype=np.float32)
            pts_i = np.array(pts_i, dtype=np.float32)

            X_ji = triangulate_between_cams(R_j, t_j, R_i, t_i, pts_j, pts_i, K)

            # Safety: NaN/Inf cleanup immediately after triangulation
            valid_tr = np.isfinite(X_ji).all(axis=1)
            if not np.all(valid_tr):
                removed = len(X_ji) - np.count_nonzero(valid_tr)
                print(f"[WARN] Triangulation ({j},{i}): removing {removed} invalid points")
            X_ji = X_ji[valid_tr]
            pts_j_g = pts_j[valid_tr]
            pts_i_g = pts_i[valid_tr]
            matches_g = [m for m, keep in zip(match_indices, valid_tr) if keep]

            if X_ji.shape[0] == 0:
                continue

            # Cheirality
            X_cam_j = (R_j @ X_ji.T + t_j.reshape(3, 1)).T
            X_cam_i = (R_i @ X_ji.T + t_i.reshape(3, 1)).T
            good = (X_cam_j[:, 2] > 0) & (X_cam_i[:, 2] > 0)
            if not np.any(good):
                continue

            X_ji = X_ji[good]
            pts_j_g = pts_j_g[good]
            pts_i_g = pts_i_g[good]
            matches_g = [m for m, keep in zip(matches_g, good) if keep]

            if X_ji.shape[0] == 0:
                continue

            # Baseline angle
            ang = triangulation_angles(R_j, t_j, R_i, t_i, X_ji)
            ang_good = (ang > MIN_TRIANGULATION_ANGLE)
            if not np.any(ang_good):
                continue

            X_ji = X_ji[ang_good]
            pts_j_g = pts_j_g[ang_good]
            pts_i_g = pts_i_g[ang_good]
            matches_g = [m for m, keep in zip(matches_g, ang_good) if keep]

            if X_ji.shape[0] == 0:
                continue

            # Safety again before reprojection
            valid_tr2 = np.isfinite(X_ji).all(axis=1)
            if not np.all(valid_tr2):
                removed2 = len(X_ji) - np.count_nonzero(valid_tr2)
                print(f"[WARN] Triangulation ({j},{i}) post-angle: removing {removed2} invalid points")
            X_ji = X_ji[valid_tr2]
            pts_j_g = pts_j_g[valid_tr2]
            pts_i_g = pts_i_g[valid_tr2]
            matches_g = [m for m, keep in zip(matches_g, valid_tr2) if keep]

            if X_ji.shape[0] == 0:
                continue

            # Reprojection error in both views
            err_j = reprojection_errors(X_ji, R_j, t_j, K, pts_j_g)
            err_i = reprojection_errors(X_ji, R_i, t_i, K, pts_i_g)
            final_mask = (err_j < REPROJ_ERROR_TRIANG) & (err_i < REPROJ_ERROR_TRIANG)

            if not np.any(final_mask):
                continue

            X_ji = X_ji[final_mask]
            pts_j_g = pts_j_g[final_mask]
            pts_i_g = pts_i_g[final_mask]
            matches_g = [m for m, keep in zip(matches_g, final_mask) if keep]

            # Final safety
            valid_final = np.isfinite(X_ji).all(axis=1)
            if not np.all(valid_final):
                removed3 = len(X_ji) - np.count_nonzero(valid_final)
                print(f"[WARN] Triangulation ({j},{i}) post-reproj: removing {removed3} invalid points")
            X_ji = X_ji[valid_final]
            matches_g = [m for m, keep in zip(matches_g, valid_final) if keep]

            if X_ji.shape[0] == 0:
                continue

            # Add final accepted points
            for X, m in zip(X_ji, matches_g):
                p_idx = len(points3D)
                points3D.append(X)

                # Color from image j
                u, v = keypoints_list[j][m.queryIdx].pt
                u_i = int(round(u))
                v_i = int(round(v))
                u_i = np.clip(u_i, 0, wj - 1)
                v_i = np.clip(v_i, 0, hj - 1)
                color = img_j[v_i, u_i]
                colors3D.append(color)

                kp_to_3d[(j, m.queryIdx)] = p_idx
                kp_to_3d[(i, m.trainIdx)] = p_idx
                new_points += 1

        print(f"  Added {new_points} new 3D points. Total: {len(points3D)}")

    # Convert list to arrays
    points3D_arr = np.array(points3D, dtype=np.float64)
    colors3D_arr = np.array(colors3D, dtype=np.uint8)
    # ---- PRE-BA VISUALIZATION ----
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Pre-Bundle Adjustment Views", fontsize=16)

        X = points3D_arr[:, 0]
        Y = points3D_arr[:, 1]
        Z = points3D_arr[:, 2]
        C = colors3D_arr[:, ::-1] / 255.0  # BGR→RGB

        # Top view: X–Z
        ax1 = fig.add_subplot(221)
        ax1.scatter(X, Z, s=1, c=C)
        ax1.set_title("Top View (X–Z)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Z")

        # Side view: Y–Z
        ax2 = fig.add_subplot(222)
        ax2.scatter(Y, Z, s=1, c=C)
        ax2.set_title("Side View (Y–Z)")
        ax2.set_xlabel("Y")
        ax2.set_ylabel("Z")

        # Front view: X–Y
        ax3 = fig.add_subplot(223)
        ax3.scatter(X, Y, s=1, c=C)
        ax3.set_title("Front View (X–Y)")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")

        # 3D overall
        ax4 = fig.add_subplot(224, projection="3d")
        ax4.scatter(X, Y, Z, s=1, c=C)
        ax4.set_title("3D View")
        ax4.set_axis_off()

        pre_path = os.path.join(OUTPUT_DIR, "pre_BA_views.png")
        plt.savefig(pre_path, dpi=250)
        plt.show()
        print(f"[VIS] Saved pre-BA views → {pre_path}")

    except Exception as e:
        print("[VIS] Pre-BA visualization skipped:", e)

    # 6. Run bundle adjustment
    # ---- Compute initial RMS reprojection error BEFORE BA ----
    print("\n[CHECK] Computing initial RMS reprojection error (before BA)...")

    def project_point_direct(R, t, X, K):
        Xc = R @ X + t
        Z = Xc[2] if abs(Xc[2]) > 1e-9 else 1e-9
        u = K[0, 0] * (Xc[0] / Z) + K[0, 2]
        v = K[1, 1] * (Xc[1] / Z) + K[1, 2]
        return np.array([u, v])

    initial_errors = []

    for (img_idx, kp_idx), p_idx in kp_to_3d.items():
        if p_idx >= len(points3D_arr):
            continue

        R = R_list[img_idx]
        t = t_list[img_idx]
        if R is None or t is None:
            continue

        X = points3D_arr[p_idx]
        kp = keypoints_list[img_idx][kp_idx]
        u_obs, v_obs = kp.pt

        u_pred, v_pred = project_point_direct(R, t, X, K)
        err = np.sqrt((u_obs - u_pred) ** 2 + (v_obs - v_pred) ** 2)
        initial_errors.append(err)

    initial_errors = np.array(initial_errors)
    initial_rms = np.sqrt(np.mean(initial_errors ** 2)) if len(initial_errors) else float('nan')

    print(f"[CHECK] Initial reprojection RMS error: {initial_rms:.4f} px")
    print("\n[BA] Running full bundle adjustment (poses + points)...")
    points3D_arr, R_list, t_list = run_bundle_adjustment(
        points3D_arr, R_list, t_list,
        kp_to_3d, keypoints_list, K
    )
    # ---- Compute FINAL RMS reprojection error AFTER BA ----
    print("\n[CHECK] Computing FINAL RMS reprojection error (after BA)...")

    final_errors = []

    for (img_idx, kp_idx), p_idx in kp_to_3d.items():
        if p_idx >= len(points3D_arr):
            continue

        R = R_list[img_idx]
        t = t_list[img_idx]
        if R is None or t is None:
            continue

        X = points3D_arr[p_idx]
        kp = keypoints_list[img_idx][kp_idx]
        u_obs, v_obs = kp.pt

        u_pred, v_pred = project_point_direct(R, t, X, K)
        err = np.sqrt((u_obs - u_pred) ** 2 + (v_obs - v_pred) ** 2)
        final_errors.append(err)

    final_errors = np.array(final_errors)
    final_rms = np.sqrt(np.mean(final_errors ** 2)) if len(final_errors) else float('nan')

    print(f"[CHECK] FINAL reprojection RMS error: {final_rms:.4f} px")

    if np.isfinite(initial_rms) and np.isfinite(final_rms):
        improvement = 100 * (initial_rms - final_rms) / max(initial_rms, 1e-9)
        print(f"[CHECK] BA improvement: {improvement:.1f}%")

    # ---- POST-BA VISUALIZATION ----
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Post-Bundle Adjustment Views", fontsize=16)

        X = points3D_arr[:, 0]
        Y = points3D_arr[:, 1]
        Z = points3D_arr[:, 2]
        C = colors3D_arr[:, ::-1] / 255.0

        # Top view: X–Z
        ax1 = fig.add_subplot(221)
        ax1.scatter(X, Z, s=1, c=C)
        ax1.set_title("Top View (X–Z)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Z")

        # Side view: Y–Z
        ax2 = fig.add_subplot(222)
        ax2.scatter(Y, Z, s=1, c=C)
        ax2.set_title("Side View (Y–Z)")
        ax2.set_xlabel("Y")
        ax2.set_ylabel("Z")

        # Front view: X–Y
        ax3 = fig.add_subplot(223)
        ax3.scatter(X, Y, s=1, c=C)
        ax3.set_title("Front View (X–Y)")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")

        # 3D overall
        ax4 = fig.add_subplot(224, projection="3d")
        ax4.scatter(X, Y, Z, s=1, c=C)
        ax4.set_title("3D View")
        ax4.set_axis_off()

        post_path = os.path.join(OUTPUT_DIR, "post_BA_views.png")
        plt.savefig(post_path, dpi=250)
        plt.show()
        print(f"[VIS] Saved post-BA views → {post_path}")

    except Exception as e:
        print("[VIS] Post-BA visualization skipped:", e)

    # 7. Save outputs
    save_ply(points3D_arr, colors3D_arr, PLY_PATH)

    # Save camera poses that were successfully estimated
    valid_indices = []
    R_out, t_out = [], []
    for i, (R, t) in enumerate(zip(R_list, t_list)):
        if R is not None and t is not None:
            valid_indices.append(i)
            R_out.append(R)
            t_out.append(t)

    np.save(POSES_PATH, {
        "indices": np.array(valid_indices),
        "R": np.array(R_out),
        "t": np.array(t_out)
    })
    print(f"[OUTPUT] Saved camera poses: {POSES_PATH}")
    print("[DONE] Multi-view SfM + BA complete.")

    # ---- OPTIONAL 3D VISUALIZATION (Matplotlib) ----
    try:
        import matplotlib
        matplotlib.use("MacOSX")  # or another interactive backend if needed

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points3D_arr[:, 0],
            points3D_arr[:, 1],
            points3D_arr[:, 2],
            s=1,
            c=colors3D_arr[:, ::-1] / 255.0
        )
        ax.set_title("Phase 2: Multi-View SfM Point Cloud (After BA)")
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Visualization skipped:", e)


if __name__ == "__main__":
    main()
