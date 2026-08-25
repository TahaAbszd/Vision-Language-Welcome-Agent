from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel


class Keypoint(BaseModel):
    x: float
    y: float
    v: float  # visibility, 0..1


class TrackResult(BaseModel):
    id: int
    color: Tuple[int, int, int]
    keypoints: List[Keypoint]
    gestures: List[str]


class FrameResult(BaseModel):
    """One message sent back to the browser per processed frame."""

    tracks: List[TrackResult]
    server_fps: float
    processing_ms: float


class ErrorMessage(BaseModel):
    error: str
