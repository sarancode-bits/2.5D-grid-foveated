"""
sensor_fusion.py — LiDAR-Camera Sensor Fusion Engine
=====================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Phase 1 — Product Roadmap

Provides mathematically rigorous 3D LiDAR → 2D Camera projection using
calibrated extrinsic and intrinsic matrices. Replaces the monocular depth
heuristic with true metric distance from the point cloud.

Core capabilities:
  1. Load and manage calibration parameters (K, T)
  2. Project 3D LiDAR points to 2D camera pixel coordinates
  3. Associate YOLO 2D bounding boxes with LiDAR frustum points
  4. Compute true metric depth for each detected object
  5. Colorize LiDAR points with camera RGB values

Usage:
    from sensor_fusion import SensorFusionEngine

    fusion = SensorFusionEngine("calibration_config.json")
    depth = fusion.get_object_depth(lidar_points, yolo_bbox)
    projected = fusion.project_lidar_to_image(lidar_points)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from bbox3d_estimator import estimate_3d_bbox
except ImportError:
    estimate_3d_bbox = None

logger = logging.getLogger(__name__)

# Default calibration config path (same directory as this file)
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "calibration_config.json"


class CalibrationParams:
    """Container for LiDAR-Camera calibration parameters."""

    def __init__(
        self,
        intrinsic_K: np.ndarray,
        extrinsic_R: np.ndarray,
        extrinsic_t: np.ndarray,
        image_width: int = 1242,
        image_height: int = 375,
    ):
        """
        Parameters
        ----------
        intrinsic_K : np.ndarray, shape (3, 3)
            Camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1].
        extrinsic_R : np.ndarray, shape (3, 3)
            Rotation matrix from LiDAR frame to camera frame.
        extrinsic_t : np.ndarray, shape (3,)
            Translation vector from LiDAR frame to camera frame.
        image_width : int
            Camera image width in pixels.
        image_height : int
            Camera image height in pixels.
        """
        self.K = np.array(intrinsic_K, dtype=np.float64).reshape(3, 3)
        self.R = np.array(extrinsic_R, dtype=np.float64).reshape(3, 3)
        self.t = np.array(extrinsic_t, dtype=np.float64).reshape(3, 1)
        self.image_width = image_width
        self.image_height = image_height

        # Build the full 4×4 extrinsic transformation matrix T_lidar_to_cam
        self.T = np.eye(4, dtype=np.float64)
        self.T[:3, :3] = self.R
        self.T[:3, 3] = self.t.flatten()

        # Build the 3×4 projection matrix P = K @ [R | t]
        Rt = np.hstack([self.R, self.t])  # (3, 4)
        self.P = self.K @ Rt  # (3, 4)

        logger.info(
            "Calibration loaded: K[fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f], "
            "image=%dx%d",
            self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2],
            self.image_width, self.image_height,
        )

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> "CalibrationParams":
        """Load calibration from a JSON config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Calibration config not found: {config_path}")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        return cls(
            intrinsic_K=cfg["intrinsic_K"],
            extrinsic_R=cfg["extrinsic_rotation"],
            extrinsic_t=cfg["extrinsic_translation"],
            image_width=cfg.get("image_width", 1242),
            image_height=cfg.get("image_height", 375),
        )


class SensorFusionEngine:
    """LiDAR-Camera Sensor Fusion Engine.

    Projects 3D LiDAR points into the 2D camera frame and associates
    YOLO bounding box detections with their true metric depth from the
    point cloud.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        calib: Optional[CalibrationParams] = None,
    ):
        """
        Parameters
        ----------
        config_path : str | Path | None
            Path to calibration_config.json. Uses default if None.
        calib : CalibrationParams | None
            Pre-built calibration. Overrides config_path if provided.
        """
        if calib is not None:
            self.calib = calib
        else:
            path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
            self.calib = CalibrationParams.from_json(path)

    # ------------------------------------------------------------------
    # Core projection: 3D LiDAR → 2D Camera pixels
    # ------------------------------------------------------------------

    def project_lidar_to_image(
        self,
        points: np.ndarray,
        filter_behind: bool = True,
        filter_outside: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D LiDAR points to 2D camera pixel coordinates.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3) or (N, 4+)
            LiDAR XYZ[I...] point cloud in the LiDAR coordinate frame.
        filter_behind : bool
            If True, remove points behind the camera (z_cam <= 0).
        filter_outside : bool
            If True, remove points outside the image bounds.

        Returns
        -------
        pixels : np.ndarray, shape (M, 2)
            Projected pixel coordinates (u, v) for valid points.
        depths : np.ndarray, shape (M,)
            Metric depth (z in camera frame) for each valid point.
        valid_mask : np.ndarray[bool], shape (N,)
            Boolean mask over the original points indicating which
            were successfully projected into the image.
        """
        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        N = xyz.shape[0]

        # Transform to camera frame: P_cam = R @ P_lidar + t
        # Using homogeneous coordinates for efficiency
        ones = np.ones((N, 1), dtype=np.float64)
        pts_h = np.hstack([xyz, ones])  # (N, 4)

        # Project: [u*z, v*z, z] = P @ [x, y, z, 1]^T
        proj = (self.calib.P @ pts_h.T).T  # (N, 3)

        # Depth in camera frame (z_cam)
        z_cam = proj[:, 2]

        # Build validity mask
        valid = np.ones(N, dtype=bool)

        if filter_behind:
            valid &= (z_cam > 0.1)  # Must be in front of camera

        # Compute pixel coordinates (only where z > 0 to avoid div by zero)
        safe_z = np.where(z_cam > 0.1, z_cam, 1.0)
        u = proj[:, 0] / safe_z
        v = proj[:, 1] / safe_z

        if filter_outside:
            valid &= (u >= 0) & (u < self.calib.image_width)
            valid &= (v >= 0) & (v < self.calib.image_height)

        pixels = np.column_stack([u[valid], v[valid]])
        depths = z_cam[valid]

        return pixels, depths, valid

    # ------------------------------------------------------------------
    # Frustum association: 2D YOLO box → 3D LiDAR points
    # ------------------------------------------------------------------

    def get_frustum_points(
        self,
        points: np.ndarray,
        bbox_2d: Tuple[float, float, float, float],
        margin: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find all LiDAR points that project inside a 2D bounding box.

        This is the core of "frustum PointNet" style association:
        a 2D detection defines a frustum (3D cone) in LiDAR space.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3+)
            Full LiDAR point cloud.
        bbox_2d : (x1, y1, x2, y2)
            YOLO-style bounding box in pixel coordinates.
        margin : float
            Pixel margin to expand the box (catches edge points).

        Returns
        -------
        frustum_points : np.ndarray, shape (K, 3+)
            LiDAR points falling inside the frustum.
        frustum_depths : np.ndarray, shape (K,)
            Metric depth of each frustum point.
        frustum_indices : np.ndarray[int], shape (K,)
            Indices into the original points array.
        """
        x1, y1, x2, y2 = bbox_2d

        # Project all points to image
        pixels, depths, valid_mask = self.project_lidar_to_image(points)

        # Find points inside the expanded bounding box
        valid_indices = np.nonzero(valid_mask)[0]
        u, v = pixels[:, 0], pixels[:, 1]

        in_box = (
            (u >= x1 - margin) & (u <= x2 + margin) &
            (v >= y1 - margin) & (v <= y2 + margin)
        )

        frustum_mask = in_box
        frustum_indices = valid_indices[frustum_mask]
        frustum_points = points[frustum_indices]
        frustum_depths = depths[frustum_mask]

        return frustum_points, frustum_depths, frustum_indices

    def get_object_depth(
        self,
        points: np.ndarray,
        bbox_2d: Tuple[float, float, float, float],
        method: str = "median",
    ) -> Optional[float]:
        """Compute the true metric depth of a detected object.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3+)
            Full LiDAR point cloud.
        bbox_2d : (x1, y1, x2, y2)
            YOLO bounding box in pixel coordinates.
        method : str
            Depth aggregation: "median" (robust), "mean", "min" (closest).

        Returns
        -------
        depth_m : float | None
            Metric depth in metres, or None if no LiDAR points
            fall inside the frustum.
        """
        _, frustum_depths, _ = self.get_frustum_points(points, bbox_2d)

        if len(frustum_depths) == 0:
            return None

        if method == "median":
            return float(np.median(frustum_depths))
        elif method == "mean":
            return float(np.mean(frustum_depths))
        elif method == "min":
            return float(np.min(frustum_depths))
        else:
            return float(np.median(frustum_depths))

    # ------------------------------------------------------------------
    # Batch fusion: process all YOLO detections at once
    # ------------------------------------------------------------------

    def fuse_detections(
        self,
        points: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enrich a list of YOLO detections with true LiDAR metric depth.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3+)
            Full LiDAR point cloud for the current frame.
        detections : list of dict
            Each dict must contain "bbox" key with [x1, y1, x2, y2].
            Other keys (label, confidence, track_id) are preserved.

        Returns
        -------
        enriched : list of dict
            Same detections with added keys:
            - "lidar_depth_m": float | None
            - "lidar_points_in_frustum": int
            - "range_band": "near" | "mid" | "far" (from true depth)
        """
        # Pre-project all points once for efficiency
        pixels, depths, valid_mask = self.project_lidar_to_image(points)
        valid_indices = np.nonzero(valid_mask)[0]
        u_all, v_all = pixels[:, 0], pixels[:, 1]

        enriched = []
        for det in detections:
            det_copy = dict(det)
            bbox = det.get("bbox", det.get("box", [0, 0, 0, 0]))
            x1, y1, x2, y2 = bbox

            # Find points in this box
            in_box = (
                (u_all >= x1 - 5) & (u_all <= x2 + 5) &
                (v_all >= y1 - 5) & (v_all <= y2 + 5)
            )

            box_depths = depths[in_box]
            n_pts = len(box_depths)

            if n_pts > 0:
                depth_m = float(np.median(box_depths))
                det_copy["lidar_depth_m"] = round(depth_m, 2)
                det_copy["lidar_points_in_frustum"] = n_pts

                # True range band from metric depth
                if depth_m <= 10.0:
                    det_copy["range_band"] = "near"
                elif depth_m <= 30.0:
                    det_copy["range_band"] = "mid"
                else:
                    det_copy["range_band"] = "far"
                    
                # Phase 5: Estimate 3D Oriented Bounding Box for the frustum points
                if estimate_3d_bbox is not None and n_pts >= 10:
                    # In world coordinates, we might need to filter ground points first,
                    # but estimate_3d_bbox ignores Z for L-shape fitting so it's okay.
                    # Exclude ground points for height estimation if possible, but 
                    # bbox3d_estimator handles it decently.
                    # points array has [X, Y, Z]
                    pts_3d = points[valid_indices][in_box]
                    try:
                        bbox3d = estimate_3d_bbox(pts_3d)
                        if bbox3d:
                            det_copy["bbox3d"] = bbox3d.to_dict()
                    except Exception as e:
                        logger.warning(f"BBox3D estimation failed: {e}")
            else:
                det_copy["lidar_depth_m"] = None
                det_copy["lidar_points_in_frustum"] = 0
                det_copy["range_band"] = det.get("range_band", "far")

            enriched.append(det_copy)

        return enriched

    # ------------------------------------------------------------------
    # Point cloud colorization from camera image
    # ------------------------------------------------------------------

    def colorize_points(
        self,
        points: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """Paint LiDAR points with RGB color from the camera image.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3+)
            LiDAR point cloud.
        image : np.ndarray, shape (H, W, 3)
            Camera image (BGR or RGB).

        Returns
        -------
        colors : np.ndarray, shape (N, 3)
            RGB color for each point. Unprojectable points get [128, 128, 128].
        """
        N = points.shape[0]
        colors = np.full((N, 3), 128, dtype=np.uint8)

        pixels, _, valid_mask = self.project_lidar_to_image(points)
        valid_indices = np.nonzero(valid_mask)[0]

        if len(valid_indices) > 0:
            u = np.clip(pixels[:, 0].astype(int), 0, image.shape[1] - 1)
            v = np.clip(pixels[:, 1].astype(int), 0, image.shape[0] - 1)
            colors[valid_indices] = image[v, u]

        return colors


# ── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Sensor Fusion Engine — Self-Test")
    print("=" * 60)

    # Load default calibration
    fusion = SensorFusionEngine()

    # Generate synthetic LiDAR point cloud (car-like cluster at ~15m)
    rng = np.random.default_rng(42)

    # Ground plane (in front of camera: positive X in LiDAR = forward)
    ground = np.column_stack([
        rng.uniform(2, 40, 5000),
        rng.uniform(-15, 15, 5000),
        np.full(5000, -1.73),
    ])

    # Vehicle at ~15m ahead
    vehicle = np.column_stack([
        rng.uniform(14.0, 16.0, 200),
        rng.uniform(-1.0, 1.0, 200),
        rng.uniform(-1.0, 0.5, 200),
    ])

    # Pedestrian at ~8m
    person = np.column_stack([
        rng.uniform(7.5, 8.5, 50),
        rng.uniform(2.0, 2.5, 50),
        rng.uniform(-1.5, 0.0, 50),
    ])

    all_points = np.vstack([ground, vehicle, person]).astype(np.float32)

    # Project to image
    pixels, depths, valid = fusion.project_lidar_to_image(all_points)
    print(f"\n  Total points: {all_points.shape[0]:,}")
    print(f"  Projected to image: {pixels.shape[0]:,} ({valid.sum() / len(valid) * 100:.1f}%)")

    # Simulate a YOLO detection for the vehicle
    # (approximate pixel coords for a car at 15m)
    detections = [
        {"label": "car", "confidence": 0.92, "bbox": [500, 150, 700, 250]},
        {"label": "person", "confidence": 0.87, "bbox": [300, 120, 350, 220]},
    ]

    enriched = fusion.fuse_detections(all_points, detections)
    print(f"\n  Fused detections:")
    for det in enriched:
        depth_str = f"{det['lidar_depth_m']:.1f}m" if det['lidar_depth_m'] else "N/A"
        print(
            f"    {det['label']:>8s} | Depth: {depth_str:>7s} | "
            f"Range: {det['range_band']:>4s} | "
            f"LiDAR pts: {det['lidar_points_in_frustum']}"
        )

    print(f"\n  [OK] Sensor Fusion Engine verification complete!")
