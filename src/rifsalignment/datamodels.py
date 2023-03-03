"""TimedSegment class."""

from dataclasses import dataclass


@dataclass
class TimedSegment:
    start: float
    end: float
    text: str
