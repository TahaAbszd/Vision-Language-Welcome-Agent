from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .config import Settings
from .landmarks import Landmarks, dist, torso_center

TRACK_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (66, 135, 245), (245, 66, 90), (66, 245, 129),
    (245, 191, 66), (191, 66, 245), (66, 245, 233),
)


@dataclass
class PersonTrack:
    track_id: int
    history_len: int
    color: Tuple[int, int, int] = (0, 255, 0)
    last_seen_frame: int = 0

    raise_hand_streak: int = 0
    hug_streak: int = 0
    last_clap_time: float = 0.0
    last_hug_time: float = 0.0
    active_labels: Dict[str, float] = field(default_factory=dict)
    history: Deque[Landmarks] = field(init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.history_len)

    def update(self, landmarks: Landmarks) -> None:
        self.history.append(landmarks)

    def set_label(self, label: str, now: float, ttl: float) -> None:
        self.active_labels[label] = now + ttl

    def current_labels(self, now: float) -> List[str]:
        return [lbl for lbl, exp in self.active_labels.items() if exp >= now]

    @property
    def latest(self) -> Optional[Landmarks]:
        return self.history[-1] if self.history else None


class CentroidTracker:

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracks: Dict[int, PersonTrack] = {}
        self._next_id = 0
        self._frame_idx = 0

    def update(self, detections: List[Landmarks]) -> Dict[int, Landmarks]:
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
                if d <= self._settings.track_match_max_dist:
                    pairs.append((d, tid, di))
        pairs.sort(key=lambda p: p[0])

        for d, tid, di in pairs:
            if tid in assignment or di not in unmatched_dets or tid not in unmatched_tracks:
                continue
            assignment[tid] = di
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        result: Dict[int, Landmarks] = {}
        for tid, di in assignment.items():
            track = self._tracks[tid]
            track.update(detections[di])
            track.last_seen_frame = self._frame_idx
            result[tid] = detections[di]

        for di in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            track = PersonTrack(track_id=tid, history_len=self._settings.history_len,
                                 color=color, last_seen_frame=self._frame_idx)
            track.update(detections[di])
            self._tracks[tid] = track
            result[tid] = detections[di]

        stale = [
            tid for tid, t in self._tracks.items()
            if self._frame_idx - t.last_seen_frame > self._settings.track_max_missed_frames
        ]
        for tid in stale:
            del self._tracks[tid]

        return result

    @property
    def tracks(self) -> Dict[int, PersonTrack]:
        return self._tracks
