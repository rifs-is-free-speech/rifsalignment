"""
rifs alignment
"""

from rifsalignment.align_csv import align_csv
from rifsalignment.preprocess import prepare_text
from rifsalignment.datamodels import TimedSegment
from rifsalignment.algorithms import CTC

__version__ = "0.0.1"

alignment_methods = {"ctc": CTC}

__all__ = ["alignment_methods", "align_csv", "prepare_text", "TimedSegment"]
