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

@dataclass
class TimedSegmentWithModelOutput:
    """
    A timed segment of audio and text with model output.

    Parameters
    ----------
    start: float
        The start time of the segment.
    end: float
        The end time of the segment.
    text: str
        The text of the segment.
    model_output: str
        The model output for the segment.
    """

    start: float
    end: float
    text: str
    model_output: str


placeholder_text = "DETTE ER BARE EN PLADSHOLDER"
