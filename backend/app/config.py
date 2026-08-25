from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_path: str = "models/pose_landmarker_lite.task"
    max_people: int = 4
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5

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

    hug_torso_ratio: float = 1.3
    hug_wrist_reach_ratio: float = 0.9
    hug_min_frames: int = 6
    hug_cooldown_sec: float = 1.0

    label_min_display_sec: float = 0.8

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    max_process_fps: float = 20.0
    max_frame_bytes: int = 3 * 1024 * 1024


settings = Settings()
