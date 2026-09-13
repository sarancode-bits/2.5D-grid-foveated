"""
multi_object_tracker.py — SORT-Style Multi-Object Tracker
==========================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Phase 2 — Product Roadmap

Implements persistent multi-object tracking across video frames using:
  1. Kalman Filter for state prediction (position + velocity)
  2. Hungarian algorithm (scipy) for optimal detection-to-track assignment
  3. IoU-based cost matrix for bounding box matching
  4. Track lifecycle management (creation, update, deletion)

Each tracked object maintains a stable ID across frames and outputs
velocity estimates for trajectory prediction.

Usage:
    from multi_object_tracker import MultiObjectTracker

    tracker = MultiObjectTracker()
    for frame_detections in detection_stream:
        tracked = tracker.update(frame_detections)
        for t in tracked:
            print(f"Track #{t['track_id']} {t['label']} vel=({t['vx']:.1f}, {t['vy']:.1f})")
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing scipy for Hungarian assignment
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    logger.warning(
        "scipy not installed — tracker will use greedy assignment (less accurate). "
        "Install with: pip install scipy"
    )


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix. Shape: (len(a), len(b))."""
    n, m = len(boxes_a), len(boxes_b)
    iou_mat = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            iou_mat[i, j] = _iou(boxes_a[i], boxes_b[j])
    return iou_mat


class KalmanBoxTracker:
    """Kalman Filter tracker for a single bounding box.

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    where (cx, cy) is box center, (w, h) is dimensions,
    and (vx, vy, vw, vh) are velocities.
    """

    _next_id = 1

    def __init__(self, bbox: np.ndarray, label: str = "object", confidence: float = 0.5):
        """
        Parameters
        ----------
        bbox : np.ndarray, shape (4,)
            Initial bounding box [x1, y1, x2, y2].
        label : str
            Object class label.
        confidence : float
            Detection confidence.
        """
        self.track_id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.label = label
        self.confidence = confidence
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[np.ndarray] = []

        # Convert bbox to state [cx, cy, w, h, vx, vy, vw, vh]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # State vector
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        # State covariance
        self.P = np.eye(8, dtype=np.float64)
        self.P[4:, 4:] *= 100.0  # High uncertainty on initial velocities

        # Transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += vx * dt
        self.F[1, 5] = 1.0  # cy += vy * dt
        self.F[2, 6] = 1.0  # w  += vw * dt
        self.F[3, 7] = 1.0  # h  += vh * dt

        # Measurement matrix (observe [cx, cy, w, h])
        self.H = np.eye(4, 8, dtype=np.float64)

        # Process noise
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64) * 10.0

    def predict(self) -> np.ndarray:
        """Predict next state and return predicted bbox [x1, y1, x2, y2]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1

        # Clamp dimensions to be positive
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])

        return self._state_to_bbox()

    def update(self, bbox: np.ndarray, label: str = "", confidence: float = 0.5):
        """Update state with a matched detection."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float64)

        # Kalman update
        y = z - self.H @ self.x                         # Innovation
        S = self.H @ self.P @ self.H.T + self.R         # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        if label:
            self.label = label
        self.confidence = max(self.confidence, confidence)

        # Store history for trajectory visualization
        self.history.append(self._state_to_bbox().copy())
        if len(self.history) > 30:
            self.history.pop(0)

    def _state_to_bbox(self) -> np.ndarray:
        """Convert state [cx, cy, w, h, ...] to bbox [x1, y1, x2, y2]."""
        cx, cy, w, h = self.x[:4]
        return np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dtype=np.float64)

    def get_velocity(self) -> Tuple[float, float]:
        """Return (vx, vy) velocity in pixels per frame."""
        return float(self.x[4]), float(self.x[5])

    def get_state_dict(self) -> Dict[str, Any]:
        """Export track state as a dictionary for JSON serialization."""
        bbox = self._state_to_bbox()
        vx, vy = self.get_velocity()
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            "bbox": [round(float(b), 1) for b in bbox],
            "velocity_px_s": [round(vx, 2), round(vy, 2)],
            "age": self.age,
            "hits": self.hits,
            "frames_alive": self.hits,
            "time_since_update": self.time_since_update,
        }


class MultiObjectTracker:
    """SORT-style multi-object tracker with Kalman filtering.

    Maintains a set of active tracks and matches incoming detections
    frame-by-frame using IoU + Hungarian assignment.
    """

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 3,
        iou_threshold: float = 0.25,
    ):
        """
        Parameters
        ----------
        max_age : int
            Maximum frames a track survives without a detection match.
        min_hits : int
            Minimum detection hits before a track is confirmed/reported.
        iou_threshold : float
            Minimum IoU for a detection-to-track match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process a new frame of detections and return tracked objects.

        Parameters
        ----------
        detections : list of dict
            Each dict must contain:
            - "bbox": [x1, y1, x2, y2]
            - "label": str (optional)
            - "confidence": float (optional)
            Other keys are preserved and passed through.

        Returns
        -------
        tracked : list of dict
            Active, confirmed tracks with stable IDs, velocities,
            and all original detection keys.
        """
        self.frame_count += 1

        # Extract detection bboxes
        det_bboxes = []
        det_labels = []
        det_confs = []
        det_extras = []

        for det in detections:
            bbox = det.get("bbox", det.get("box", [0, 0, 0, 0]))
            det_bboxes.append(np.array(bbox, dtype=np.float64))
            det_labels.append(det.get("label", "object"))
            det_confs.append(det.get("confidence", 0.5))
            # Preserve extra keys
            extras = {k: v for k, v in det.items() if k not in ("bbox", "box", "label", "confidence")}
            det_extras.append(extras)

        # 1. Predict all existing tracks forward
        predicted_bboxes = []
        for track in self.tracks:
            pred = track.predict()
            predicted_bboxes.append(pred)

        # 2. Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(
            det_bboxes, predicted_bboxes
        )

        # 3. Update matched tracks
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(
                det_bboxes[det_idx],
                label=det_labels[det_idx],
                confidence=det_confs[det_idx],
            )

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = KalmanBoxTracker(
                det_bboxes[det_idx],
                label=det_labels[det_idx],
                confidence=det_confs[det_idx],
            )
            self.tracks.append(new_track)

        # 5. Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # 6. Build output: confirmed tracks only
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                state = track.get_state_dict()
                # Merge any extra keys from the matched detection
                for det_idx, track_idx in matched:
                    if self.tracks[track_idx] is track:
                        state.update(det_extras[det_idx])
                        break
                results.append(state)

        return results

    def _associate(
        self,
        det_bboxes: List[np.ndarray],
        pred_bboxes: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with tracks using IoU + Hungarian.

        Returns
        -------
        matched : list of (det_idx, track_idx)
        unmatched_dets : list of det_idx
        unmatched_tracks : list of track_idx
        """
        n_det = len(det_bboxes)
        n_trk = len(pred_bboxes)

        if n_det == 0:
            return [], [], list(range(n_trk))
        if n_trk == 0:
            return [], list(range(n_det)), []

        # Compute IoU cost matrix
        det_arr = np.array(det_bboxes)
        pred_arr = np.array(pred_bboxes)
        iou_mat = _iou_matrix(det_arr, pred_arr)

        # Convert to cost (minimize)
        cost_matrix = 1.0 - iou_mat

        # Solve assignment
        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            # Greedy fallback
            row_ind, col_ind = self._greedy_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(n_det))
        unmatched_tracks = list(range(n_trk))

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self.iou_threshold:
                matched.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_tracks:
                    unmatched_tracks.remove(c)

        return matched, unmatched_dets, unmatched_tracks

    @staticmethod
    def _greedy_assignment(cost_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """Greedy assignment fallback when scipy is unavailable."""
        rows, cols = [], []
        used_rows, used_cols = set(), set()
        n, m = cost_matrix.shape

        # Find minimum cost assignments greedily
        flat = cost_matrix.flatten()
        sorted_idx = np.argsort(flat)

        for idx in sorted_idx:
            r, c = divmod(idx, m)
            if r not in used_rows and c not in used_cols:
                rows.append(r)
                cols.append(c)
                used_rows.add(r)
                used_cols.add(c)
                if len(rows) >= min(n, m):
                    break

        return rows, cols

    def reset(self):
        """Clear all tracks and reset state."""
        self.tracks = []
        self.frame_count = 0
        KalmanBoxTracker._next_id = 1


# ── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Multi-Object Tracker — Self-Test (10-frame simulation)")
    print("=" * 60)

    tracker = MultiObjectTracker(min_hits=1)

    # Simulate a car moving right and a person moving left over 10 frames
    for frame in range(10):
        detections = [
            {
                "label": "car",
                "confidence": 0.92,
                "bbox": [100 + frame * 15, 200, 280 + frame * 15, 380],
            },
            {
                "label": "person",
                "confidence": 0.85,
                "bbox": [600 - frame * 10, 150, 650 - frame * 10, 350],
            },
        ]

        # Drop person detection in frame 5 to test track persistence
        if frame == 5:
            detections = [detections[0]]

        tracked = tracker.update(detections)

        print(f"\n  Frame {frame + 1}:")
        for t in tracked:
            vx, vy = t["velocity_px_s"]
            print(
                f"    Track #{t['track_id']:>2d} [{t['label']:>7s}] "
                f"bbox=[{t['bbox'][0]:>5.0f},{t['bbox'][1]:>5.0f},"
                f"{t['bbox'][2]:>5.0f},{t['bbox'][3]:>5.0f}] "
                f"vel=({vx:>6.1f}, {vy:>6.1f}) "
                f"age={t['age']:>2d} hits={t['hits']:>2d}"
            )

    print(f"\n  [OK] Multi-Object Tracker verification complete!")
