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
from rifsalignment.experiment import align_experiment_folder

__version__ = "0.2.1"

alignment_methods = {
    "ctc": CTC,
    "StateMachineForLevenshtein": StateMachineForLevenshtein,
    "StateMachineUnsupervised": StateMachineUnsupervised,
    "StateMachineUnsupervisedNoModel": StateMachineUnsupervisedNoModel,
}

__all__ = ["alignment_methods", "align_csv", "TimedSegment", "align_experiment_folder"]
