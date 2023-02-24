"""
rifs alignment
"""

from rifsalignment.align_csv import align_csv

__version__ = "0.0.1"

alignment_methods = {"ctc": None}

__all__ = ["alignment_methods", "align_csv"]
