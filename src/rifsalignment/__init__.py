"""
rifs alignment
"""

from rifsalignment.align_csv import align_csv
from rifsalignment.prepare_text import prepare_text

__version__ = "0.0.1"

alignment_methods = {"ctc": None}

__all__ = ["alignment_methods", "align_csv", "prepare_text"]
