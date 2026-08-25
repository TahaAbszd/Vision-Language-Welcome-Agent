from __future__ import annotations

from app.config import Settings
from app.tracker import CentroidTracker

from .fake_pose import make_pose


def test_same_person_keeps_id_across_small_movement():
    settings = Settings(track_match_max_dist=0.15)
    tracker = CentroidTracker(settings)

    frame1 = [make_pose()]
    matched1 = tracker.update(frame1)
    assert len(matched1) == 1
    first_id = next(iter(matched1))

    nudged = make_pose({
        11: (0.41, 0.3, 1.0), 12: (0.61, 0.3, 1.0),
        23: (0.43, 0.55, 1.0), 24: (0.59, 0.55, 1.0),
    })
    matched2 = tracker.update([nudged])
    assert len(matched2) == 1
    assert first_id in matched2


def test_new_detection_far_away_gets_new_id():
    settings = Settings(track_match_max_dist=0.05)
    tracker = CentroidTracker(settings)

    tracker.update([make_pose()])
    far = {
        11: (0.9, 0.3, 1.0), 12: (1.1, 0.3, 1.0),
        23: (0.92, 0.55, 1.0), 24: (1.08, 0.55, 1.0),
        13: (0.85, 0.4, 1.0), 14: (1.15, 0.4, 1.0),
        15: (0.9, 0.5, 1.0), 16: (1.1, 0.5, 1.0),
    }
    matched = tracker.update([make_pose(far)])
    assert len(tracker.tracks) == 2
    assert len(matched) == 1


def test_stale_track_is_dropped_after_missed_frames():
    settings = Settings(track_match_max_dist=0.15, track_max_missed_frames=2)
    tracker = CentroidTracker(settings)

    tracker.update([make_pose()])
    assert len(tracker.tracks) == 1

    tracker.update([]) 
    tracker.update([]) 
    tracker.update([])  
    assert len(tracker.tracks) == 0


def test_two_simultaneous_people_get_distinct_ids():
    settings = Settings(track_match_max_dist=0.1)
    tracker = CentroidTracker(settings)

    person_b = {
        11: (1.4, 0.3, 1.0), 12: (1.6, 0.3, 1.0),
        23: (1.42, 0.55, 1.0), 24: (1.58, 0.55, 1.0),
        13: (1.35, 0.4, 1.0), 14: (1.65, 0.4, 1.0),
        15: (1.4, 0.5, 1.0), 16: (1.6, 0.5, 1.0),
    }
    matched = tracker.update([make_pose(), make_pose(person_b)])
    assert len(matched) == 2
    assert len(tracker.tracks) == 2
