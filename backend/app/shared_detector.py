from __future__ import annotations

import asyncio
from typing import List

import numpy as np

from .config import Settings
from .pose_backend import build_detector, detect_poses


class SharedPoseDetector:
    """Wraps the single mediapipe PoseLandmarker instance for the process.

    mediapipe's Python Task objects are not documented as safe for
    concurrent calls from multiple threads, so every call is serialized
    behind a lock while the actual inference runs in a worker thread (via
    asyncio.to_thread) so it never blocks the event loop for other
    connections. For higher throughput than one lock can provide, scale
    out with multiple worker processes behind a load balancer instead of
    removing this lock.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detector = build_detector(settings)
        self._lock = asyncio.Lock()

    async def detect(self, bgr_frame: np.ndarray) -> List:
        async with self._lock:
            return await asyncio.to_thread(detect_poses, self._detector, bgr_frame)

    def close(self) -> None:
        self._detector.close()
