
from __future__ import annotations

from typing import Protocol, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


class Landmark(Protocol):
    x: float
    y: float
    visibility: float | None


Landmarks = Sequence[Landmark]


def to_xy(landmarks: Landmarks, idx: int) -> Point:
    lm = landmarks[idx]
    return (lm.x, lm.y)


def dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def shoulder_width(landmarks: Landmarks) -> float:
    return max(dist(to_xy(landmarks, L_SHOULDER), to_xy(landmarks, R_SHOULDER)), 1e-4)


def torso_center(landmarks: Landmarks) -> Point:
    pts = [to_xy(landmarks, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    xs, ys = zip(*pts)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def is_visible(landmarks: Landmarks, idx: int, thresh: float = 0.4) -> bool:
    lm = landmarks[idx]
    vis = getattr(lm, "visibility", 1.0)
    return vis is None or vis >= thresh
