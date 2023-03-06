"""TimedSegment class."""

from dataclasses import dataclass


@dataclass
class TimedSegment:
    """
    A timed segment of audio and text.

    Parameters
    ----------
    start: float
        The start time of the segment.
    end: float
        The end time of the segment.
    text: str
        The text of the segment.
    """

    start: float
    end: float
    text: str


placeholder_text = "DETTE ER BARE EN PLADSHOLDER"
