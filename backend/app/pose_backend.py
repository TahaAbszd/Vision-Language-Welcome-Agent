from __future__ import annotations

import logging
import os

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as exc: 
    raise RuntimeError(
        "mediapipe is not installed. Run: pip install -r requirements.txt"
    ) from exc

from .config import Settings

log = logging.getLogger("pose_backend")


class ModelNotFoundError(RuntimeError):
    pass


def ensure_model_present(path: str) -> None:
    if os.path.exists(path):
        return
    raise ModelNotFoundError(
        f"Pose model file not found at '{path}'. Download it once with:\n"
        f"  curl -L -o {path} "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task\n"
        "(swap '_lite' for '_full' or '_heavy' for higher accuracy at the "
        "cost of speed)"
    )


def build_detector(settings: Settings) -> mp_vision.PoseLandmarker:

    ensure_model_present(settings.model_path)
    base_options = mp_python.BaseOptions(
        model_asset_path=settings.model_path,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=settings.max_people,
        min_pose_detection_confidence=settings.min_detection_confidence,
        min_pose_presence_confidence=settings.min_presence_confidence,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def detect_poses(detector: mp_vision.PoseLandmarker, bgr_frame: np.ndarray):
    """Runs pose detection on one BGR frame (as decoded by cv2.imdecode)."""
    rgb = bgr_frame[:, :, ::-1]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_image)
    return result.pose_landmarks or []
