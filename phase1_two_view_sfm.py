import os
import glob
import numpy as np
import cv2


# --------- CONFIG ---------
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
PLY_PATH = os.path.join(OUTPUT_DIR, "phase1_point_cloud.ply")
# --------------------------
import matplotlib
matplotlib.use("MacOSX")  # or "Qt5Agg"

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_two_images(data_dir: str):
    """Load two images from Data/ (first two sorted .jpg)."""
    pattern = os.path.join(data_dir, "*.jpg")
    files = sorted(glob.glob(pattern))
    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 images in {data_dir}, found {len(files)}")

    img1_path, img2_path = files[0], files[1]
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise RuntimeError("Failed to read one of the images.")

    print(f"Using images:\n  {os.path.basename(img1_path)}\n  {os.path.basename(img2_path)}")
    return img1, img2


def approximate_intrinsics(image):
    """
    Approximate camera intrinsics:
    fx = fy = image width
    cx, cy = image center
    """
    h, w = image.shape[:2]
    fx = fy = float(w)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    print("Approximated intrinsics K:\n", K)
    return K


def create_sift_detector():
    """Create SIFT detector (requires opencv-contrib-python)."""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    else:
        return cv2.xfeatures2d.SIFT_create()


def detect_and_match(img1_gray, img2_gray, ratio=0.75):
    """
    Detect SIFT and match with Lowe's ratio test.
    Returns:
        pts1, pts2: Nx2 float32 pixel coordinates of matched points.
        matches_good: list of cv2.DMatch
        kps1, kps2: keypoints lists (for optional visualization)
    """
    sift = create_sift_detector()

    kps1, desc1 = sift.detectAndCompute(img1_gray, None)
    kps2, desc2 = sift.detectAndCompute(img2_gray, None)

    if desc1 is None or desc2 is None:
        raise RuntimeError("No SIFT features found in one of the images.")

    print(f"Image 1: {len(kps1)} keypoints")
    print(f"Image 2: {len(kps2)} keypoints")

    # FLANN matcher for SIFT (float descriptors)
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)

    knn_matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    print(f"Matches after Lowe ratio test ({ratio}): {len(good_matches)}")

    if len(good_matches) < 8:
        raise RuntimeError("Not enough matches for Essential matrix estimation (need >= 8).")

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, good_matches, kps1, kps2


def estimate_essential(pts1, pts2, K):
    """
    Estimate Essential matrix with RANSAC.
    pts1, pts2: Nx2 pixel coords
    """
    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        raise RuntimeError("cv2.findEssentialMat failed.")
    inliers = mask.ravel().astype(bool)
    num_inliers = np.count_nonzero(inliers)
    print(f"Essential matrix estimated. Inliers: {num_inliers}/{len(pts1)}")

    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    return E, pts1_in, pts2_in


def decompose_essential_to_poses(E):
    """
    Decompose Essential matrix E into four possible (R, t) pairs.
    Returns list of (R, t) candidates.
    """
    U, S, Vt = np.linalg.svd(E)
    # Enforce det(U) > 0, det(V) > 0
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]  # third column of U

    # Fix rotations to have det(R) = +1
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


def triangulate_points(R, t, pts1_norm, pts2_norm):
    """
    Triangulate points for a given pose (R, t).
    pts*_norm: Nx2 normalized coordinates (camera frame).
    Returns:
        X: Nx3 3D points in first camera frame.
    """
    # Projection matrices for normalized coordinates
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])       # [I | 0]
    P1 = np.hstack([R, t.reshape(3, 1)])                # [R | t]

    pts1 = pts1_norm.T  # 2xN
    pts2 = pts2_norm.T  # 2xN

    X_h = cv2.triangulatePoints(P0, P1, pts1, pts2)     # 4xN
    X = (X_h[:3, :] / X_h[3, :]).T                      # Nx3
    return X


def cheirality_check(R, t, X):
    """
    Count how many points are in front of both cameras.
    X: Nx3 in first camera frame (world = cam0).
    """
    # In camera 0 frame (identity): just check Z > 0
    z0 = X[:, 2]

    # In camera 1 frame: X1 = R*X + t
    X1 = (R @ X.T + t.reshape(3, 1)).T
    z1 = X1[:, 2]

    in_front = (z0 > 0) & (z1 > 0)
    count = np.count_nonzero(in_front)
    return count, in_front


def choose_best_pose(E, pts1, pts2, K):
    """
    From E and the inlier points, pick the correct (R, t) via cheirality.
    Returns:
        R_best, t_best, X_best (3D points), mask_in_front (bool)
    """
    # Normalize pixels to camera coordinates
    pts1_undist = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts2_undist = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    poses = decompose_essential_to_poses(E)

    best_count = -1
    best = None

    for idx, (R, t) in enumerate(poses):
        X = triangulate_points(R, t, pts1_undist, pts2_undist)
        count, in_front = cheirality_check(R, t, X)
        print(f"Pose {idx}: {count} points in front of both cameras")
        if count > best_count:
            best_count = count
            best = (R, t, X, in_front)

    R_best, t_best, X_best, mask_front = best
    print(f"Selected pose with {best_count} points in front of both cameras.")
    return R_best, t_best, X_best, mask_front


def save_ply(points_3d, colors_bgr, filepath):
    """
    Save 3D points (Nx3) + colors (Nx3 BGR) to an ASCII PLY file.
    Colors will be stored as RGB.
    """
    ensure_output_dir()
    points = np.asarray(points_3d, dtype=np.float32)
    colors = np.asarray(colors_bgr, dtype=np.uint8)
    assert points.shape[0] == colors.shape[0]

    n_points = points.shape[0]

    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            # c is BGR in OpenCV; flip to RGB
            r, g, b = int(c[2]), int(c[1]), int(c[0])
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")

    print(f"Saved point cloud to {filepath}")


def main():
    ensure_output_dir()

    # 1. Load images
    img1_color, img2_color = load_two_images(DATA_DIR)
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # 2. Approximate intrinsics
    K = approximate_intrinsics(img1_color)

    # 3. Detect + match features
    pts1, pts2, matches, kps1, kps2 = detect_and_match(img1_gray, img2_gray, ratio=0.75)

    # Optional: visualize matches
    match_vis = cv2.drawMatches(
        img1_color, kps1,
        img2_color, kps2,
        matches[:200], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "phase1_matches.jpg"), match_vis)
    print("Saved match visualization to outputs/phase1_matches.jpg")

    # 4. Estimate Essential matrix with RANSAC
    E, pts1_in, pts2_in = estimate_essential(pts1, pts2, K)

    # 5. Cheirality-based pose disambiguation
    R, t, X, mask_front = choose_best_pose(E, pts1_in, pts2_in, K)

    # Keep only points that are in front of both cameras
    X_valid = X[mask_front]

    # 6. Assign colors from first image to 3D points (using corresponding pixels)
    pts1_in_front = pts1_in[mask_front]
    # Round and clamp pixel indices
    h, w = img1_color.shape[:2]
    u = np.clip(np.round(pts1_in_front[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(pts1_in_front[:, 1]).astype(int), 0, h - 1)
    colors = img1_color[v, u]

    # 7. Save point cloud as PLY
    save_ply(X_valid, colors, PLY_PATH)

    print("Phase 1 complete: two-view reconstruction PLY saved.")


    # ---- OPTIONAL VISUALIZATION ----
    try:
        import matplotlib

        matplotlib.use("MacOSX")  # ensure interactive backend

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_valid[:, 0], X_valid[:, 1], X_valid[:, 2],
                   s=1, c=colors[:, ::-1] / 255.0)

        ax.set_title("Phase 1: Two-View 3D Point Cloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Visualization skipped:", e)


if __name__ == "__main__":
    main()
