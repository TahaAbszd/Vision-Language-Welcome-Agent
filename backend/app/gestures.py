
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

from .config import Settings
from .landmarks import (
    L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST,
    dist, is_visible, shoulder_width, to_xy, torso_center,
)
from .tracker import PersonTrack

log = logging.getLogger("gestures")


def detect_raise_hand(track: PersonTrack, settings: Settings) -> Optional[str]:
    lm = track.latest
    if lm is None:
        return None

    raised = False
    for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        if not (is_visible(lm, wrist_idx) and is_visible(lm, shoulder_idx)):
            continue
        if lm[wrist_idx].y < lm[shoulder_idx].y - settings.raise_hand_margin:
            raised = True
            break

    track.raise_hand_streak = track.raise_hand_streak + 1 if raised else 0
    if track.raise_hand_streak >= settings.raise_hand_min_frames:
        return "raise_hand"
    return None


def _wrist_x_series(track: PersonTrack, wrist_idx: int, n: int) -> List[float]:
    frames = list(track.history)[-n:]
    return [f[wrist_idx].x for f in frames if is_visible(f, wrist_idx)]


def detect_wave(track: PersonTrack, settings: Settings) -> Optional[str]:
    lm = track.latest
    if lm is None or len(track.history) < settings.wave_window:
        return None
    sw = shoulder_width(lm)

    for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        if not is_visible(lm, wrist_idx):
            continue
        if lm[wrist_idx].y >= lm[shoulder_idx].y - settings.raise_hand_margin:
            continue

        xs = _wrist_x_series(track, wrist_idx, settings.wave_window)
        if len(xs) < settings.wave_window * 0.7:
            continue

        amplitude = (max(xs) - min(xs)) / sw
        if amplitude < settings.wave_min_amplitude_ratio:
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

        if reversals >= settings.wave_min_reversals:
            return "wave"
    return None


def detect_clap(track: PersonTrack, now: float, settings: Settings) -> Optional[str]:
    if now - track.last_clap_time < settings.clap_cooldown_sec:
        return None
    frames = list(track.history)[-settings.clap_window:]
    if len(frames) < settings.clap_window:
        return None

    dists: List[Optional[float]] = []
    for f in frames:
        if not (is_visible(f, L_WRIST) and is_visible(f, R_WRIST)):
            dists.append(None)
            continue
        sw = shoulder_width(f)
        dists.append(dist(to_xy(f, L_WRIST), to_xy(f, R_WRIST)) / sw)

    valid = [d for d in dists if d is not None]
    if len(valid) < settings.clap_window * 0.7:
        return None

    current = dists[-1]
    if current is None or current > settings.clap_close_ratio:
        return None

    was_open = any(d is not None and d > settings.clap_open_ratio for d in dists[:-3])
    if was_open:
        track.last_clap_time = now
        return "clap"
    return None


@dataclass
class _PairState:
    streak: int = 0
    last_time: float = 0.0


class HugDetector:


    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pairs: Dict[FrozenSet[int], _PairState] = {}

    def update(self, tracks: Dict[int, PersonTrack], now: float) -> Dict[int, str]:
        s = self._settings
        needed = (L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        ids = [tid for tid, t in tracks.items() if t.latest is not None]
        result: Dict[int, str] = {}
        seen_pairs: set[FrozenSet[int]] = set()

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                pair_key = frozenset((a_id, b_id))
                seen_pairs.add(pair_key)
                state = self._pairs.setdefault(pair_key, _PairState())

                a, b = tracks[a_id].latest, tracks[b_id].latest
                if not (all(is_visible(a, k) for k in needed)
                        and all(is_visible(b, k) for k in needed)):
                    state.streak = 0
                    continue

                avg_sw = (shoulder_width(a) + shoulder_width(b)) / 2.0
                center_a, center_b = torso_center(a), torso_center(b)
                close = dist(center_a, center_b) / avg_sw <= s.hug_torso_ratio

                def reaches(src, dst_center) -> bool:
                    return any(
                        dist(to_xy(src, w), dst_center) / avg_sw <= s.hug_wrist_reach_ratio
                        for w in (L_WRIST, R_WRIST)
                    )

                reach = reaches(a, center_b) or reaches(b, center_a)
                state.streak = state.streak + 1 if (close and reach) else 0

                if (state.streak >= s.hug_min_frames
                        and now - state.last_time >= s.hug_cooldown_sec):
                    state.last_time = now
                    state.streak = 0
                    result[a_id] = "hug"
                    result[b_id] = "hug"

        for key in list(self._pairs):
            if key not in seen_pairs:
                del self._pairs[key]

        return result
