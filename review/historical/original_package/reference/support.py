"""Physical support utilities; interval hull never fills an unobserved gap."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Support:
    intervals: tuple[tuple[float,float], ...]
    roi_xyxy: tuple[float,float,float,float] = (0.,0.,1.,1.)

    def __post_init__(self):
        last = -math.inf
        for a,b in self.intervals:
            if not (math.isfinite(a) and math.isfinite(b) and a < b and a >= last):
                raise ValueError("support must be sorted non-overlapping finite intervals")
            last=b
        x1,y1,x2,y2=self.roi_xyxy
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("invalid normalized ROI")

    def contains(self, t: float) -> bool:
        return any(a <= t < b for a,b in self.intervals)

    def distance(self, t: float) -> float:
        if not self.intervals:return math.inf
        return min(max(a-t,0.0,t-b) for a,b in self.intervals)

    @property
    def observed_duration(self) -> float:
        return sum(b-a for a,b in self.intervals)
