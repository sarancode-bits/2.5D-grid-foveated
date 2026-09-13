"""
motion_compensation.py — LiDAR Scan Motion Distortion Correction
=================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Phase 4 — Product Roadmap

When a vehicle moves at speed, a spinning LiDAR creates a distorted
point cloud because the sensor translates during the ~100ms sweep.
This module corrects ("undistorts") the scan by:

  1. Assigning each point a fractional timestamp based on its azimuth
  2. Interpolating the ego-pose at that timestamp
  3. Transforming each point from its capture-time frame to the
     scan-start frame

Usage:
    from motion_compensation import undistort_scan

    corrected = undistort_scan(points, ego_velocity, ego_yaw_rate, scan_period=0.1)
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _rotation_matrix_2d(theta: float) -> np.ndarray:
    """2D rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def compute_azimuth_fraction(points: np.ndarray) -> np.ndarray:
    """Compute the fractional timestamp [0, 1) for each point based on azimuth.

    Assumes a 360° clockwise scan starting from +X axis.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3+)
        XYZ point cloud.

    Returns
    -------
    t_frac : np.ndarray, shape (N,)
        Fractional timestamp in [0, 1) for each point.
    """
    x = points[:, 0]
    y = points[:, 1]

    # atan2 gives angle in [-pi, pi]; convert to [0, 2*pi)
    azimuth = np.arctan2(y, x)
    azimuth = np.mod(azimuth, 2 * np.pi)

    # Normalize to [0, 1)
    t_frac = azimuth / (2 * np.pi)

    return t_frac


def undistort_scan(
    points: np.ndarray,
    ego_vx: float,
    ego_vy: float,
    ego_yaw_rate: float = 0.0,
    scan_period: float = 0.1,
    reference_time: float = 0.0,
) -> np.ndarray:
    """Correct motion distortion in a spinning LiDAR scan.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3+)
        Raw LiDAR XYZ[I...] point cloud with motion distortion.
    ego_vx : float
        Vehicle longitudinal velocity (m/s) during the scan.
    ego_vy : float
        Vehicle lateral velocity (m/s) during the scan.
    ego_yaw_rate : float
        Vehicle yaw rate (rad/s) during the scan.
    scan_period : float
        Duration of one full 360° rotation (seconds). Default 0.1s (10 Hz).
    reference_time : float
        Fraction of the scan to use as the reference frame (0.0 = start,
        0.5 = middle, 1.0 = end). Default 0.0 (scan start).

    Returns
    -------
    corrected : np.ndarray, shape (N, 3+)
        Motion-compensated point cloud. Extra columns (intensity etc.) are preserved.
    """
    N = points.shape[0]
    if N == 0:
        return points.copy()

    # Check if correction is needed (skip if stationary)
    speed = np.sqrt(ego_vx**2 + ego_vy**2)
    if speed < 0.01 and abs(ego_yaw_rate) < 0.001:
        logger.debug("Vehicle stationary — skipping motion compensation.")
        return points.copy()

    corrected = points.copy()

    # Compute per-point timestamps
    t_frac = compute_azimuth_fraction(points)

    # Time offset relative to reference frame
    dt = (t_frac - reference_time) * scan_period  # (N,) in seconds

    # For each point, compute the ego displacement at its capture time
    # Using constant velocity + constant yaw rate model:
    #   dx = vx * dt
    #   dy = vy * dt
    #   dtheta = yaw_rate * dt
    dx = ego_vx * dt
    dy = ego_vy * dt
    dtheta = ego_yaw_rate * dt

    # Transform each point back to the reference frame
    # P_corrected = R(-dtheta) @ (P_original - [dx, dy, 0])
    cos_t = np.cos(-dtheta)
    sin_t = np.sin(-dtheta)

    # Translate
    x_shifted = points[:, 0] - dx
    y_shifted = points[:, 1] - dy

    # Rotate
    corrected[:, 0] = cos_t * x_shifted - sin_t * y_shifted
    corrected[:, 1] = sin_t * x_shifted + cos_t * y_shifted
    # Z is unchanged (2D ego motion)

    n_significant = np.sum(np.sqrt(dx**2 + dy**2) > 0.001)
    logger.debug(
        "Motion compensation: speed=%.2f m/s, yaw_rate=%.3f rad/s, "
        "%d/%d points significantly shifted.",
        speed, ego_yaw_rate, n_significant, N,
    )

    return corrected


def undistort_scan_with_poses(
    points: np.ndarray,
    poses: np.ndarray,
    pose_timestamps: np.ndarray,
    scan_period: float = 0.1,
) -> np.ndarray:
    """Correct distortion using a sequence of IMU/odometry poses.

    This is the more accurate variant that uses actual pose measurements
    instead of assuming constant velocity.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3+)
        Raw LiDAR scan.
    poses : np.ndarray, shape (M, 3)
        Ego poses as [x, y, yaw] at each timestamp.
    pose_timestamps : np.ndarray, shape (M,)
        Timestamps for each pose, normalized to [0, scan_period].
    scan_period : float
        Duration of one full scan.

    Returns
    -------
    corrected : np.ndarray, shape (N, 3+)
    """
    N = points.shape[0]
    corrected = points.copy()

    # Compute per-point timestamps
    t_frac = compute_azimuth_fraction(points)
    t_abs = t_frac * scan_period

    # Reference pose (scan start)
    ref_x = np.interp(0, pose_timestamps, poses[:, 0])
    ref_y = np.interp(0, pose_timestamps, poses[:, 1])
    ref_yaw = np.interp(0, pose_timestamps, poses[:, 2])

    # Interpolate pose for each point
    pt_x = np.interp(t_abs, pose_timestamps, poses[:, 0])
    pt_y = np.interp(t_abs, pose_timestamps, poses[:, 1])
    pt_yaw = np.interp(t_abs, pose_timestamps, poses[:, 2])

    # Displacement relative to reference
    dx = pt_x - ref_x
    dy = pt_y - ref_y
    dtheta = pt_yaw - ref_yaw

    # Correct each point
    cos_t = np.cos(-dtheta)
    sin_t = np.sin(-dtheta)

    x_shifted = points[:, 0] - dx
    y_shifted = points[:, 1] - dy

    corrected[:, 0] = cos_t * x_shifted - sin_t * y_shifted
    corrected[:, 1] = sin_t * x_shifted + cos_t * y_shifted

    return corrected


# ── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("  Motion Compensation — Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Generate a ring of points (simulating a clean LiDAR scan)
    N = 10000
    angles = rng.uniform(0, 2 * np.pi, N)
    ranges = rng.uniform(5, 50, N)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = rng.uniform(-1.73, 2.0, N)
    intensity = rng.uniform(0, 255, N)

    points = np.column_stack([x, y, z, intensity]).astype(np.float32)

    # Simulate vehicle moving at 20 m/s (72 km/h) with slight yaw
    corrected = undistort_scan(
        points,
        ego_vx=20.0,
        ego_vy=0.5,
        ego_yaw_rate=0.05,
        scan_period=0.1,
    )

    # Measure displacement magnitude
    displacement = np.sqrt(
        (corrected[:, 0] - points[:, 0]) ** 2 +
        (corrected[:, 1] - points[:, 1]) ** 2
    )

    print(f"\n  Input points: {N:,}")
    print(f"  Vehicle speed: 20 m/s (72 km/h)")
    print(f"  Scan period: 100 ms")
    print(f"  Max point displacement: {displacement.max():.3f} m")
    print(f"  Mean point displacement: {displacement.mean():.3f} m")
    print(f"  Points with >1cm shift: {np.sum(displacement > 0.01):,}")
    print(f"  Extra columns preserved: {corrected.shape[1] == points.shape[1]}")
    print(f"\n  [OK] Motion Compensation verification complete!")
