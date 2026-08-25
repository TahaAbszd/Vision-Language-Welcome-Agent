from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with an env var
    of the same name (case-insensitive), e.g. MAX_PEOPLE=6 or a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Pose model
    model_path: str = "models/pose_landmarker_lite.task"
    max_people: int = 4
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5

    # Per-session tracking (state kept in server memory, per WebSocket connection)
    history_len: int = 30
    track_match_max_dist: float = 0.15
    track_max_missed_frames: int = 15

    # Gesture thresholds (unit: normalized image coordinates unless noted)
    raise_hand_margin: float = 0.03
    raise_hand_min_frames: int = 5

    wave_window: int = 15
    wave_min_reversals: int = 2
    wave_min_amplitude_ratio: float = 0.35

    clap_close_ratio: float = 0.45
    clap_open_ratio: float = 1.0
    clap_window: int = 10
    clap_cooldown_sec: float = 0.6

    hug_torso_ratio: float = 1.3
    hug_wrist_reach_ratio: float = 0.9
    hug_min_frames: int = 6
    hug_cooldown_sec: float = 1.0

    label_min_display_sec: float = 0.8

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    # Max frames/sec processed per connection; extra frames from a fast
    # client are dropped so the server never falls behind in real time.
    max_process_fps: float = 20.0
    # JPEG frames larger than this are rejected (denial-of-service guard).
    max_frame_bytes: int = 3 * 1024 * 1024


settings = Settings()
