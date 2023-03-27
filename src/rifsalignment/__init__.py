"""
rifs alignment
"""

from rifsalignment.align_csv import align_csv
from rifsalignment.preprocess import prepare_text
from rifsalignment.datamodels import TimedSegment
from rifsalignment.algorithms import (
    CTC,
    StateMachineForLevenshtein,
    StateMachineUnsupervised,
)

__version__ = "0.0.1"

alignment_methods = {
    "ctc": CTC,
    "StateMachineForLevenshtein": StateMachineForLevenshtein,
    "StateMachineUnsupervised": StateMachineUnsupervised,
}

__all__ = ["alignment_methods", "align_csv", "prepare_text", "TimedSegment"]
