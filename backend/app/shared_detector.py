from __future__ import annotations

import asyncio
from typing import List

import numpy as np

from .config import Settings
from .pose_backend import build_detector, detect_poses


class SharedPoseDetector:


    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detector = build_detector(settings)
        self._lock = asyncio.Lock()

    async def detect(self, bgr_frame: np.ndarray) -> List:
        async with self._lock:
            return await asyncio.to_thread(detect_poses, self._detector, bgr_frame)

    def close(self) -> None:
        self._detector.close()
