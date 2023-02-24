"""
Base classes for the rifs alignment library.
"""

from abc import ABC, abstractmethod


class BaseAlignment(ABC):
    """
    Base class for all alignment algorithms.
    """

    @staticmethod
    @abstractmethod
    def align(
        audio: str,
        text: str,
        model: str,
    ):
        """
        Align the source and target audio files.

        Parameters
        ----------
        audio: str
            The path to the source audio wav file.
        text: str
            The path to the source text file.
        model: str
            The path to the model to use for alignment. Can be a huggingface model or a local path.
        """
        ...