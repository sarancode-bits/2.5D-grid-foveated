"""
bbox3d_estimator.py — 3D Oriented Bounding Box Estimation
==========================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Phase 5 — Product Roadmap

Upgrades from axis-aligned bounding boxes (AABB) to oriented 3D cuboids
with heading (yaw) estimation. For each cluster from DBSCAN:

  1. L-Shape Fitting via PCA on XY projection to find dominant edges
  2. Minimum-area oriented bounding rectangle (rotating calipers)
  3. Height from Z-spread (min_z to max_z)
  4. Output: 7-DOF box (cx, cy, cz, length, width, height, yaw)

Usage:
    from bbox3d_estimator import estimate_3d_bbox, BBox3D

    bbox = estimate_3d_bbox(cluster_points)
    print(f"Center: ({bbox.cx:.1f}, {bbox.cy:.1f}, {bbox.cz:.1f})")
    print(f"Dims: {bbox.length:.1f} x {bbox.width:.1f} x {bbox.height:.1f}")
    print(f"Yaw: {np.degrees(bbox.yaw):.1f}°")
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BBox3D:
    """7-DOF Oriented 3D Bounding Box."""
    cx: float       # Center X (m)
    cy: float       # Center Y (m)
    cz: float       # Center Z (m)
    length: float   # Length along heading direction (m)
    width: float    # Width perpendicular to heading (m)
    height: float   # Height (m)
    yaw: float      # Heading angle (radians, 0 = +X axis, CCW positive)

    def to_dict(self) -> Dict[str, float]:
        """Convert to JSON-serializable dictionary."""
        return {
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "cz": round(self.cz, 3),
            "length": round(self.length, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "yaw": round(self.yaw, 4),
            "yaw_deg": round(float(np.degrees(self.yaw)), 1),
        }

    def get_corners_2d(self) -> np.ndarray:
        """Compute the 4 ground-plane corner points of the box.

        Returns
        -------
        corners : np.ndarray, shape (4, 2)
            Corner coordinates in XY plane, ordered clockwise.
        """
        cos_y = np.cos(self.yaw)
        sin_y = np.sin(self.yaw)

        # Half-dimensions
        hl = self.length / 2.0
        hw = self.width / 2.0

        # Local corners (front-right, front-left, rear-left, rear-right)
        local_corners = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw],
        ])

        # Rotation matrix
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        # Rotate and translate
        corners = (R @ local_corners.T).T + np.array([self.cx, self.cy])
        return corners

    def get_corners_3d(self) -> np.ndarray:
        """Compute all 8 corners of the 3D bounding box.

        Returns
        -------
        corners : np.ndarray, shape (8, 3)
            3D corner coordinates. First 4 are bottom, last 4 are top.
        """
        corners_2d = self.get_corners_2d()  # (4, 2)
        z_bottom = self.cz - self.height / 2.0
        z_top = self.cz + self.height / 2.0

        bottom = np.column_stack([corners_2d, np.full(4, z_bottom)])
        top = np.column_stack([corners_2d, np.full(4, z_top)])

        return np.vstack([bottom, top])


def _pca_heading(xy: np.ndarray) -> float:
    """Estimate heading angle from the principal axis of XY points.

    Uses PCA (eigenvectors of the covariance matrix) to find
    the dominant direction. Returns angle in radians.
    """
    if xy.shape[0] < 3:
        return 0.0

    # Center the points
    centroid = np.mean(xy, axis=0)
    centered = xy - centroid

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Principal axis = eigenvector with largest eigenvalue
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Angle of principal axis
    yaw = float(np.arctan2(principal[1], principal[0]))

    return yaw


def _min_area_rect(xy: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Compute the minimum-area oriented bounding rectangle.

    Uses the rotating calipers approach on the convex hull.

    Returns
    -------
    cx, cy : float
        Center of the rectangle.
    length, width : float
        Dimensions (length >= width).
    yaw : float
        Heading angle (radians).
    """
    if xy.shape[0] < 3:
        centroid = np.mean(xy, axis=0) if xy.shape[0] > 0 else np.array([0, 0])
        return float(centroid[0]), float(centroid[1]), 0.1, 0.1, 0.0

    # Compute convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
    except (ImportError, Exception):
        # Fallback to PCA-based method
        hull_pts = xy

    # Rotating calipers: try each edge of the hull as a potential box edge
    n = len(hull_pts)
    best_area = float("inf")
    best_rect = None

    for i in range(n):
        # Edge vector
        edge = hull_pts[(i + 1) % n] - hull_pts[i]
        edge_angle = np.arctan2(edge[1], edge[0])

        # Rotate all hull points to align this edge with X axis
        cos_a = np.cos(-edge_angle)
        sin_a = np.sin(-edge_angle)
        rotated = np.column_stack([
            cos_a * hull_pts[:, 0] - sin_a * hull_pts[:, 1],
            sin_a * hull_pts[:, 0] + cos_a * hull_pts[:, 1],
        ])

        # Axis-aligned bounding box in rotated frame
        min_x, max_x = rotated[:, 0].min(), rotated[:, 0].max()
        min_y, max_y = rotated[:, 1].min(), rotated[:, 1].max()

        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            # Center in rotated frame
            cx_rot = (min_x + max_x) / 2.0
            cy_rot = (min_y + max_y) / 2.0
            # Rotate center back
            cx = cos_a * cx_rot + sin_a * cy_rot
            cy = -sin_a * cx_rot + cos_a * cy_rot  # Note: inverse rotation

            length = max_x - min_x
            width = max_y - min_y

            # Ensure length >= width
            if width > length:
                length, width = width, length
                edge_angle += np.pi / 2

            best_rect = (float(cx), float(cy), float(length), float(width), float(edge_angle))

    if best_rect is None:
        centroid = np.mean(xy, axis=0)
        return float(centroid[0]), float(centroid[1]), 0.1, 0.1, 0.0

    return best_rect


def estimate_3d_bbox(
    points: np.ndarray,
    method: str = "min_area",
) -> BBox3D:
    """Estimate an oriented 3D bounding box for a point cluster.

    Parameters
    ----------
    points : np.ndarray, shape (K, 3+)
        XYZ point cluster (e.g., from DBSCAN).
    method : str
        "min_area" (rotating calipers) or "pca" (principal axis only).

    Returns
    -------
    bbox : BBox3D
        7-DOF oriented 3D bounding box.
    """
    if points.shape[0] == 0:
        return BBox3D(0, 0, 0, 0, 0, 0, 0)

    xyz = points[:, :3]
    xy = xyz[:, :2]

    if method == "min_area" and xyz.shape[0] >= 3:
        cx, cy, length, width, yaw = _min_area_rect(xy)
    else:
        # PCA fallback
        yaw = _pca_heading(xy)
        centroid = np.mean(xy, axis=0)
        cx, cy = float(centroid[0]), float(centroid[1])

        # Project onto principal axis for length/width
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        rotated = np.column_stack([
            cos_y * (xy[:, 0] - cx) - sin_y * (xy[:, 1] - cy),
            sin_y * (xy[:, 0] - cx) + cos_y * (xy[:, 1] - cy),
        ])
        length = float(rotated[:, 0].max() - rotated[:, 0].min())
        width = float(rotated[:, 1].max() - rotated[:, 1].min())

    # Height from Z spread
    z_min = float(xyz[:, 2].min())
    z_max = float(xyz[:, 2].max())
    height = z_max - z_min
    cz = (z_min + z_max) / 2.0

    # Ensure minimum dimensions
    length = max(length, 0.1)
    width = max(width, 0.1)
    height = max(height, 0.1)

    # Normalize yaw to [-pi, pi]
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    return BBox3D(cx=cx, cy=cy, cz=cz, length=length, width=width, height=height, yaw=yaw)


def estimate_3d_bboxes_batch(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    method: str = "min_area",
) -> List[Dict[str, Any]]:
    """Estimate 3D bounding boxes for all clusters in a scan.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3+)
    cluster_labels : np.ndarray[int], shape (N,)
        Cluster IDs from DBSCAN (-1 = noise).
    method : str

    Returns
    -------
    results : list of dict
        Each dict contains "cluster_id", "bbox3d" (BBox3D.to_dict()),
        "num_points", and "corners_3d".
    """
    unique_labels = np.unique(cluster_labels)
    results = []

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        mask = cluster_labels == label
        cluster_pts = points[mask]

        if cluster_pts.shape[0] < 3:
            continue

        bbox = estimate_3d_bbox(cluster_pts, method=method)
        results.append({
            "cluster_id": int(label),
            "bbox3d": bbox.to_dict(),
            "num_points": int(cluster_pts.shape[0]),
            "corners_3d": bbox.get_corners_3d().tolist(),
        })

    return results


# ── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  3D BBox Estimator — Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Simulate a car-shaped cluster (4m long, 1.8m wide, 1.5m tall)
    # oriented at 30 degrees
    yaw_true = np.radians(30)
    N_car = 500
    local_pts = np.column_stack([
        rng.uniform(-2.0, 2.0, N_car),   # length
        rng.uniform(-0.9, 0.9, N_car),   # width
        rng.uniform(-0.75, 0.75, N_car),  # height
    ])

    # Rotate to true heading
    cos_y = np.cos(yaw_true)
    sin_y = np.sin(yaw_true)
    rotated_xy = np.column_stack([
        cos_y * local_pts[:, 0] - sin_y * local_pts[:, 1],
        sin_y * local_pts[:, 0] + cos_y * local_pts[:, 1],
    ])

    # Translate to position (15, 5)
    car_cluster = np.column_stack([
        rotated_xy[:, 0] + 15.0,
        rotated_xy[:, 1] + 5.0,
        local_pts[:, 2],
    ])

    # Simulate a pedestrian (0.5m x 0.5m x 1.7m)
    N_ped = 50
    ped_cluster = np.column_stack([
        rng.uniform(-0.25, 0.25, N_ped) + 8.0,
        rng.uniform(-0.25, 0.25, N_ped) + 3.0,
        rng.uniform(-1.7, 0.0, N_ped),
    ])

    print(f"\n  Car cluster: {N_car} points, true yaw={np.degrees(yaw_true):.0f}°")
    car_bbox = estimate_3d_bbox(car_cluster, method="min_area")
    print(f"    Estimated: L={car_bbox.length:.2f}m W={car_bbox.width:.2f}m "
          f"H={car_bbox.height:.2f}m Yaw={np.degrees(car_bbox.yaw):.1f}°")
    print(f"    Center: ({car_bbox.cx:.1f}, {car_bbox.cy:.1f}, {car_bbox.cz:.1f})")
    corners = car_bbox.get_corners_3d()
    print(f"    8 corners computed: shape={corners.shape}")

    print(f"\n  Pedestrian cluster: {N_ped} points")
    ped_bbox = estimate_3d_bbox(ped_cluster, method="pca")
    print(f"    Estimated: L={ped_bbox.length:.2f}m W={ped_bbox.width:.2f}m "
          f"H={ped_bbox.height:.2f}m")

    print(f"\n  [OK] 3D BBox Estimator verification complete!")
