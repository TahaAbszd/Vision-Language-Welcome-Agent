
from __future__ import annotations

from typing import Dict, List, Tuple

NUM_LANDMARKS = 33


class FakeLandmark:
    def __init__(self, x: float, y: float, visibility: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


def make_pose(overrides: Dict[int, Tuple[float, float, float]] | None = None) -> List[FakeLandmark]:

    base = [FakeLandmark(0.1, 0.1) for _ in range(NUM_LANDMARKS)]
    defaults = {
        11: (0.4, 0.3, 1.0),  
        12: (0.6, 0.3, 1.0),  
        13: (0.35, 0.4, 1.0), 
        14: (0.65, 0.4, 1.0),  
        15: (0.4, 0.5, 1.0), 
        16: (0.6, 0.5, 1.0),  
        23: (0.42, 0.55, 1.0), 
        24: (0.58, 0.55, 1.0), 
    }
    defaults.update(overrides or {})
    for idx, (x, y, v) in defaults.items():
        base[idx] = FakeLandmark(x, y, v)
    return base
