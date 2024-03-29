"""
Base classes for the rifs alignment library.
"""

from abc import ABC, abstractmethod

from rifsalignment.datamodels import TimedSegment

from typing import List


class BaseAlignment(ABC):
    """
    Base class for all alignment algorithms.
    """

    @staticmethod
    @abstractmethod
    def align(
        audio_file: str,
        text_file: str,
        model: str,
        verbose: bool = False,
        quiet: bool = False,
        **kwargs,
    ) -> List[TimedSegment]:
        """
        Align the source and target audio files.

        Parameters
        ----------
        audio_file: str
            The path to the source audio wav file.
        text_file: str
            The path to the source text file.
        model: str
            The path to the model to use for alignment. Can be a huggingface model or a local path.
        verbose: bool
            Whether to print the alignments progress with steps.
        quiet: bool
            Prints nothing.
        """
        ...
