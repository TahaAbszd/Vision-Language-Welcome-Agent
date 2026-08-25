"""Minimal stand-in for mediapipe's NormalizedLandmark list, so gesture and
tracker logic can be unit tested without installing mediapipe/opencv.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

NUM_LANDMARKS = 33


class FakeLandmark:
    def __init__(self, x: float, y: float, visibility: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


def make_pose(overrides: Dict[int, Tuple[float, float, float]] | None = None) -> List[FakeLandmark]:
    """33 landmarks defaulted to a relaxed standing pose, with specific
    indices overridden as {index: (x, y, visibility)}.

    Defaults: shoulders at y=0.3 (11 at x=0.4, 12 at x=0.6), hips at
    y=0.55, wrists at y=0.5 (down by the hips), everything else parked
    off to the side so it doesn't interfere with distance checks.
    """
    base = [FakeLandmark(0.1, 0.1) for _ in range(NUM_LANDMARKS)]
    defaults = {
        11: (0.4, 0.3, 1.0),  # left shoulder
        12: (0.6, 0.3, 1.0),  # right shoulder
        13: (0.35, 0.4, 1.0),  # left elbow
        14: (0.65, 0.4, 1.0),  # right elbow
        15: (0.4, 0.5, 1.0),  # left wrist
        16: (0.6, 0.5, 1.0),  # right wrist
        23: (0.42, 0.55, 1.0),  # left hip
        24: (0.58, 0.55, 1.0),  # right hip
    }
    defaults.update(overrides or {})
    for idx, (x, y, v) in defaults.items():
        base[idx] = FakeLandmark(x, y, v)
    return base
