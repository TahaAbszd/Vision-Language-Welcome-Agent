from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as exc:
    raise SystemExit(
        "mediapipe is not installed. Run: pip install -r requirements.txt"
    ) from exc


@dataclass
class Config:
    model_path: str = "pose_landmarker_lite.task"
    max_people: int = 4
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540

    history_len: int = 30

    track_match_max_dist: float = 0.15
    track_max_missed_frames: int = 15

    raise_hand_margin: float = 0.03
    raise_hand_min_frames: int = 5

    wave_window: int = 15
    wave_min_reversals: int = 2
    wave_min_amplitude_ratio: float = 0.35

    clap_close_ratio: float = 0.45
    clap_open_ratio: float = 1.0
    clap_window: int = 10
    clap_cooldown_sec: float = 0.6

    # "hug" = one person opening both arms wide toward the camera.
    hug_wrist_span_ratio: float = 2.0   # wrist-to-wrist span >= this * shoulder width
    hug_open_margin: float = 0.02       # each wrist this far outside its shoulder (norm. x)
    hug_height_slack: float = 0.4       # wrists allowed this * torso height above shoulders
    hug_min_frames: int = 6
    hug_cooldown_sec: float = 1.0

    label_min_display_sec: float = 0.8

    debug_hug: bool = True


CFG = Config()

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

TRACK_COLORS = [
    (66, 135, 245), (245, 66, 90), (66, 245, 129),
    (245, 191, 66), (191, 66, 245), (66, 245, 233),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gesture_app")


Point = Tuple[float, float]


def to_xy(landmarks, idx: int) -> Point:
    lm = landmarks[idx]
    return (lm.x, lm.y)


def dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def shoulder_width(landmarks) -> float:
    return max(dist(to_xy(landmarks, L_SHOULDER), to_xy(landmarks, R_SHOULDER)), 1e-4)


def torso_center(landmarks) -> Point:
    pts = [to_xy(landmarks, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    xs, ys = zip(*pts)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def is_visible(landmarks, idx: int, thresh: float = 0.4) -> bool:
    lm = landmarks[idx]
    vis = getattr(lm, "visibility", 1.0)
    return vis is None or vis >= thresh


@dataclass
class PersonTrack:
    track_id: int
    history: Deque[object] = field(default_factory=lambda: deque(maxlen=CFG.history_len))
    color: Tuple[int, int, int] = (0, 255, 0)
    last_seen_frame: int = 0

    raise_hand_streak: int = 0
    hug_streak: int = 0
    last_clap_time: float = 0.0
    last_hug_time: float = 0.0
    active_labels: Dict[str, float] = field(default_factory=dict)

    def update(self, landmarks) -> None:
        self.history.append(landmarks)

    def set_label(self, label: str, now: float) -> None:
        self.active_labels[label] = now + CFG.label_min_display_sec

    def current_labels(self, now: float) -> List[str]:
        return [lbl for lbl, exp in self.active_labels.items() if exp >= now]

    @property
    def latest(self):
        return self.history[-1] if self.history else None


class SimpleCentroidTracker:
    def __init__(self) -> None:
        self._tracks: Dict[int, PersonTrack] = {}
        self._next_id = 0
        self._frame_idx = 0

    def update(self, detections: List) -> Dict[int, object]:
        self._frame_idx += 1
        det_centers = [torso_center(d) for d in detections]

        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))
        assignment: Dict[int, int] = {}

        pairs = []
        for tid, track in self._tracks.items():
            if track.latest is None:
                continue
            tc = torso_center(track.latest)
            for di, dc in enumerate(det_centers):
                d = dist(tc, dc)
                if d <= CFG.track_match_max_dist:
                    pairs.append((d, tid, di))
        pairs.sort(key=lambda p: p[0])

        for d, tid, di in pairs:
            if tid in assignment or di not in unmatched_dets:
                continue
            if tid not in unmatched_tracks:
                continue
            assignment[tid] = di
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        result: Dict[int, object] = {}
        for tid, di in assignment.items():
            track = self._tracks[tid]
            track.update(detections[di])
            track.last_seen_frame = self._frame_idx
            result[tid] = detections[di]

        for di in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            track = PersonTrack(track_id=tid, color=color, last_seen_frame=self._frame_idx)
            track.update(detections[di])
            self._tracks[tid] = track
            result[tid] = detections[di]

        stale = [
            tid for tid, t in self._tracks.items()
            if self._frame_idx - t.last_seen_frame > CFG.track_max_missed_frames
        ]
        for tid in stale:
            del self._tracks[tid]

        return result

    @property
    def tracks(self) -> Dict[int, PersonTrack]:
        return self._tracks


def detect_raise_hand(track: PersonTrack) -> Optional[str]:
    lm = track.latest
    if lm is None:
        return None
    sw = shoulder_width(lm)
    raised = False
    for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        if not (is_visible(lm, wrist_idx) and is_visible(lm, shoulder_idx)):
            continue
        wrist_y = lm[wrist_idx].y
        shoulder_y = lm[shoulder_idx].y
        if wrist_y < shoulder_y - CFG.raise_hand_margin:
            raised = True
            break

    if raised:
        track.raise_hand_streak += 1
    else:
        track.raise_hand_streak = 0

    if track.raise_hand_streak >= CFG.raise_hand_min_frames:
        return "raise_hand"
    return None


def _wrist_x_series(track: PersonTrack, wrist_idx: int, n: int) -> List[float]:
    frames = list(track.history)[-n:]
    return [f[wrist_idx].x for f in frames if is_visible(f, wrist_idx)]


def detect_wave(track: PersonTrack) -> Optional[str]:
    lm = track.latest
    if lm is None or len(track.history) < CFG.wave_window:
        return None
    sw = shoulder_width(lm)

    for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        if not is_visible(lm, wrist_idx):
            continue
        if lm[wrist_idx].y >= lm[shoulder_idx].y - CFG.raise_hand_margin:
            continue

        xs = _wrist_x_series(track, wrist_idx, CFG.wave_window)
        if len(xs) < CFG.wave_window * 0.7:
            continue

        amplitude = (max(xs) - min(xs)) / sw
        if amplitude < CFG.wave_min_amplitude_ratio:
            continue

        reversals = 0
        direction = 0
        for a, b in zip(xs, xs[1:]):
            step = b - a
            if abs(step) < 1e-4:
                continue
            new_dir = 1 if step > 0 else -1
            if direction != 0 and new_dir != direction:
                reversals += 1
            direction = new_dir

        if reversals >= CFG.wave_min_reversals:
            return "wave"
    return None


def detect_clap(track: PersonTrack, now: float) -> Optional[str]:
    if now - track.last_clap_time < CFG.clap_cooldown_sec:
        return None
    frames = list(track.history)[-CFG.clap_window:]
    if len(frames) < CFG.clap_window:
        return None

    dists = []
    for f in frames:
        if not (is_visible(f, L_WRIST) and is_visible(f, R_WRIST)):
            dists.append(None)
            continue
        sw = shoulder_width(f)
        d = dist(to_xy(f, L_WRIST), to_xy(f, R_WRIST)) / sw
        dists.append(d)

    valid = [d for d in dists if d is not None]
    if len(valid) < CFG.clap_window * 0.7:
        return None

    current = dists[-1]
    if current is None or current > CFG.clap_close_ratio:
        return None

    was_open = any(d is not None and d > CFG.clap_open_ratio for d in dists[:-3])
    if was_open:
        track.last_clap_time = now
        return "clap"
    return None


def detect_hug(track: PersonTrack, now: float) -> Optional[str]:
    """One person opening both arms wide, as if to hug the camera."""
    lm = track.latest
    if lm is None:
        return None

    if now - track.last_hug_time < CFG.hug_cooldown_sec:
        return None

    needed = (L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if not all(is_visible(lm, i) for i in needed):
        track.hug_streak = 0
        return None

    sw = shoulder_width(lm)
    l_wrist, r_wrist = to_xy(lm, L_WRIST), to_xy(lm, R_WRIST)
    l_sh, r_sh = to_xy(lm, L_SHOULDER), to_xy(lm, R_SHOULDER)

    span = dist(l_wrist, r_wrist) / sw
    wide = span >= CFG.hug_wrist_span_ratio
    open_out = (
        l_wrist[0] > l_sh[0] + CFG.hug_open_margin
        and r_wrist[0] < r_sh[0] - CFG.hug_open_margin
    )

    shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
    hip_y = (to_xy(lm, L_HIP)[1] + to_xy(lm, R_HIP)[1]) / 2.0
    torso_h = max(hip_y - shoulder_y, 1e-4)
    top = shoulder_y - CFG.hug_height_slack * torso_h
    at_height = (
        top <= l_wrist[1] <= hip_y and top <= r_wrist[1] <= hip_y
    )

    ok = wide and open_out and at_height

    if CFG.debug_hug:
        log.info(
            "HUG? id=%d span/sw=%.2f (thr>=%.2f) wide=%s open_out=%s "
            "at_height=%s streak=%d",
            track.track_id, span, CFG.hug_wrist_span_ratio, wide,
            open_out, at_height, track.hug_streak,
        )

    if not ok:
        track.hug_streak = 0
        return None

    track.hug_streak += 1
    if track.hug_streak >= CFG.hug_min_frames:
        track.last_hug_time = now
        track.hug_streak = 0
        return "hug"
    return None


def draw_skeleton(frame: np.ndarray, landmarks, color: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def draw_track_label(frame: np.ndarray, track: PersonTrack, now: float) -> None:
    lm = track.latest
    if lm is None:
        return
    h, w = frame.shape[:2]
    nose_xy = to_xy(lm, NOSE)
    x, y = int(nose_xy[0] * w), int(nose_xy[1] * h) - 20

    labels = track.current_labels(now)
    text = f"ID {track.track_id}" + (f" | {' + '.join(labels)}" if labels else "")
    cv2.putText(frame, text, (max(x, 5), max(y, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, track.color, 2, cv2.LINE_AA)


def ensure_model_present(path: str) -> None:
    if os.path.exists(path):
        return
    log.error(
        "Pose model file not found at '%s'.\n"
        "Download it once with:\n"
        "  curl -L -o %s "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task\n"
        "(Use 'pose_landmarker_full' or '_heavy' instead of '_lite' for higher "
        "accuracy at the cost of speed.)",
        path, path,
    )
    raise SystemExit(1)


def build_detector(cfg: Config) -> mp_vision.PoseLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=cfg.model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=cfg.max_people,
        min_pose_detection_confidence=cfg.min_detection_confidence,
        min_pose_presence_confidence=cfg.min_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def main() -> None:
    ensure_model_present(CFG.model_path)
    detector = build_detector(CFG)
    tracker = SimpleCentroidTracker()

    cap = cv2.VideoCapture(CFG.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.frame_height)
    if not cap.isOpened():
        log.error("Could not open camera index %d.", CFG.camera_index)
        sys.exit(1)

    log.info("Camera opened. Press 'q' to quit.")
    start_time = time.time()
    fps_smooth = 0.0
    prev_tick = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("Failed to read frame from camera; stopping.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            detections = result.pose_landmarks or []

            matched = tracker.update(detections)
            now = time.time()

            for tid, landmarks in matched.items():
                track = tracker.tracks[tid]
                draw_skeleton(frame, landmarks, track.color)

                for label in (
                    detect_raise_hand(track),
                    detect_wave(track),
                    detect_clap(track, now),
                    detect_hug(track, now),
                ):
                    if label:
                        track.set_label(label, now)

                draw_track_label(frame, track, now)

            tick = time.time()
            inst_fps = 1.0 / max(tick - prev_tick, 1e-6)
            fps_smooth = fps_smooth * 0.9 + inst_fps * 0.1
            prev_tick = tick
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        log.info("Shut down cleanly.")


if __name__ == "__main__":
    main()
