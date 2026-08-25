from __future__ import annotations

import time

import numpy as np

from .config import Settings
from .gestures import HugDetector, detect_clap, detect_raise_hand, detect_wave
from .schemas import FrameResult, Keypoint, TrackResult
from .shared_detector import SharedPoseDetector
from .tracker import CentroidTracker


class GestureSession:

    def __init__(self, settings: Settings, shared_detector: SharedPoseDetector) -> None:
        self._settings = settings
        self._detector = shared_detector
        self._tracker = CentroidTracker(settings)
        self._hug_detector = HugDetector(settings)
        self._fps_smooth = 0.0
        self._prev_tick = time.time()

    async def process_frame(self, bgr_frame: np.ndarray) -> FrameResult:
        start = time.time()
        detections = await self._detector.detect(bgr_frame)
        matched = self._tracker.update(detections)
        now = time.time()

        hug_labels = self._hug_detector.update(self._tracker.tracks, now)

        tracks: list[TrackResult] = []
        for tid, landmarks in matched.items():
            track = self._tracker.tracks[tid]

            fired = [
                detect_raise_hand(track, self._settings),
                detect_wave(track, self._settings),
                detect_clap(track, now, self._settings),
                hug_labels.get(tid),
            ]
            for label in fired:
                if label:
                    track.set_label(label, now, self._settings.label_min_display_sec)

            tracks.append(
                TrackResult(
                    id=tid,
                    color=track.color,
                    keypoints=[
                        Keypoint(x=lm.x, y=lm.y, v=getattr(lm, "visibility", 1.0) or 1.0)
                        for lm in landmarks
                    ],
                    gestures=track.current_labels(now),
                )
            )

        tick = time.time()
        inst_fps = 1.0 / max(tick - self._prev_tick, 1e-6)
        self._fps_smooth = self._fps_smooth * 0.9 + inst_fps * 0.1
        self._prev_tick = tick

        return FrameResult(
            tracks=tracks,
            server_fps=round(self._fps_smooth, 1),
            processing_ms=round((time.time() - start) * 1000, 1),
        )
