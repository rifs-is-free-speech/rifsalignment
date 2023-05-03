"""
rifs alignment
"""

from rifsalignment.align_csv import align_csv
from rifsalignment.datamodels import TimedSegment
from rifsalignment.algorithms import (
    CTC,
    StateMachineForLevenshtein,
    StateMachineUnsupervised,
    StateMachineUnsupervisedNoModel,
)

__version__ = "0.2.0"

alignment_methods = {
    "ctc": CTC,
    "StateMachineForLevenshtein": StateMachineForLevenshtein,
    "StateMachineUnsupervised": StateMachineUnsupervised,
    "StateMachineUnsupervisedNoModel": StateMachineUnsupervisedNoModel,
}

__all__ = ["alignment_methods", "align_csv", "TimedSegment"]
