from __future__ import annotations

from app.config import Settings
from app.gestures import HugDetector, detect_clap, detect_raise_hand, detect_wave
from app.tracker import PersonTrack

from .fake_pose import make_pose


def test_raise_hand_needs_sustained_frames():
    settings = Settings(raise_hand_min_frames=3, raise_hand_margin=0.03)
    track = PersonTrack(track_id=0, history_len=30)
    pose = make_pose({15: (0.4, 0.2, 1.0)})  # left wrist raised above left shoulder
    track.update(pose)

    assert detect_raise_hand(track, settings) is None
    assert detect_raise_hand(track, settings) is None
    assert detect_raise_hand(track, settings) == "raise_hand"


def test_raise_hand_resets_when_lowered():
    settings = Settings(raise_hand_min_frames=2, raise_hand_margin=0.03)
    track = PersonTrack(track_id=0, history_len=30)

    track.update(make_pose({15: (0.4, 0.2, 1.0)}))
    detect_raise_hand(track, settings)
    track.update(make_pose({15: (0.4, 0.5, 1.0)}))  # wrist back down
    assert detect_raise_hand(track, settings) is None
    assert track.raise_hand_streak == 0


def test_wave_detects_oscillating_raised_wrist():
    settings = Settings(wave_window=6, wave_min_reversals=2, wave_min_amplitude_ratio=0.3)
    track = PersonTrack(track_id=0, history_len=30)

    xs = [0.3, 0.5, 0.3, 0.5, 0.3, 0.5]
    for x in xs:
        track.update(make_pose({15: (x, 0.2, 1.0)}))

    assert detect_wave(track, settings) == "wave"


def test_wave_requires_enough_history():
    settings = Settings(wave_window=6)
    track = PersonTrack(track_id=0, history_len=30)
    track.update(make_pose({15: (0.3, 0.2, 1.0)}))
    assert detect_wave(track, settings) is None


def test_clap_fires_after_opening_then_closing():
    settings = Settings(clap_window=5, clap_close_ratio=0.5, clap_open_ratio=1.0, clap_cooldown_sec=0.0)
    track = PersonTrack(track_id=0, history_len=30)

    frames = [
        (0.0, 1.0),   # wide open
        (0.1, 0.9),
        (0.2, 0.8),
        (0.35, 0.65),
        (0.49, 0.51),  # hands together
    ]
    for lx, rx in frames:
        track.update(make_pose({15: (lx, 0.5, 1.0), 16: (rx, 0.5, 1.0)}))

    assert detect_clap(track, now=1.0, settings=settings) == "clap"


def test_clap_respects_cooldown():
    settings = Settings(clap_window=5, clap_close_ratio=0.5, clap_open_ratio=1.0, clap_cooldown_sec=5.0)
    track = PersonTrack(track_id=0, history_len=30)
    frames = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.35, 0.65), (0.49, 0.51)]
    for lx, rx in frames:
        track.update(make_pose({15: (lx, 0.5, 1.0), 16: (rx, 0.5, 1.0)}))

    assert detect_clap(track, now=10.0, settings=settings) == "clap"
    # Same close hands again immediately after: cooldown should suppress it.
    track.update(make_pose({15: (0.49, 0.5, 1.0), 16: (0.51, 0.5, 1.0)}))
    assert detect_clap(track, now=10.1, settings=settings) is None


def test_hug_fires_for_close_pair_with_reaching_wrist():
    settings = Settings(hug_torso_ratio=1.5, hug_wrist_reach_ratio=1.0,
                         hug_min_frames=2, hug_cooldown_sec=0.0)
    hug = HugDetector(settings)

    track_a = PersonTrack(track_id=0, history_len=30)
    track_a.update(make_pose())  # default pose, torso centered near x=0.5

    dx = 0.15
    shifted = {
        11: (0.4 + dx, 0.3, 1.0), 12: (0.6 + dx, 0.3, 1.0),
        13: (0.35 + dx, 0.4, 1.0), 14: (0.65 + dx, 0.4, 1.0),
        15: (0.4 + dx, 0.5, 1.0), 16: (0.6 + dx, 0.5, 1.0),
        23: (0.42 + dx, 0.55, 1.0), 24: (0.58 + dx, 0.55, 1.0),
    }
    track_b = PersonTrack(track_id=1, history_len=30)
    track_b.update(make_pose(shifted))

    tracks = {0: track_a, 1: track_b}
    assert hug.update(tracks, now=1.0) == {}  # streak = 1, not enough yet
    result = hug.update(tracks, now=1.0)      # streak = 2, fires
    assert result == {0: "hug", 1: "hug"}


def test_hug_does_not_fire_for_distant_pair():
    settings = Settings(hug_torso_ratio=1.5, hug_wrist_reach_ratio=1.0, hug_min_frames=1)
    hug = HugDetector(settings)

    track_a = PersonTrack(track_id=0, history_len=30)
    track_a.update(make_pose())

    far = {k: (v[0] + 3.0, v[1], v[2]) for k, v in {
        11: (0.4, 0.3, 1.0), 12: (0.6, 0.3, 1.0),
        13: (0.35, 0.4, 1.0), 14: (0.65, 0.4, 1.0),
        15: (0.4, 0.5, 1.0), 16: (0.6, 0.5, 1.0),
        23: (0.42, 0.55, 1.0), 24: (0.58, 0.55, 1.0),
    }.items()}
    track_b = PersonTrack(track_id=1, history_len=30)
    track_b.update(make_pose(far))

    result = hug.update({0: track_a, 1: track_b}, now=1.0)
    assert result == {}
