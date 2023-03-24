"""
Main way of interacting with this package.
"""

from rifsalignment.base import BaseAlignment
from rifsalignment.datamodels import placeholder_text, TimedSegment

from typing import List

import pandas as pd
import os


def align_csv(
    data_path: str,
    align_method: BaseAlignment,
    model: str,
    target_path: str = None,
    verbose: bool = False,
    quiet: bool = False,
):
    """
    Adapter fucntion. Primarily serves to make rifsalignment compatible with the rifs CLI.

    Parameters
    ----------
    data_path: str
        The path to the csv file containing the audio and text files to align. Should of the rifs format.
    align_method: BaseAlignment
        The method to use for alignment. Can be one of the following: CTC
    model: str
        The path to the model to use for alignment. Can be a huggingface model or a local path.
    target_path: str
        The path to the folder to save the aligned files to.
        If None, the files will be saved to the same folder as the csv file.
    verbose: bool
        Whether to print the alignments progress with steps.
    quiet: bool
        Prints nothing.
    """

    # Load csv file
    all_csv = pd.read_csv(os.path.join(data_path, "all.csv"), header=0)

    if target_path is None:
        target_path = os.path.join(data_path, "alignments")
        os.makedirs(target_path, exist_ok=True)

    # Retrieve audio and text files and parse to align
    for i, row in all_csv.iterrows():
        if verbose and not quiet:
            print(f"Aligning {row['id']}")

        alignments = align_method.align(
            audio_file=os.path.join(data_path, "audio", row["id"] + ".wav"),
            text_file=os.path.join(data_path, "text", row["id"] + ".txt"),
            model=model,
            verbose=verbose,
            quiet=quiet,
            # TODO: Parse kwargs from environment variables
        )

        save_alignments(target_path=target_path, id=row["id"], alignments=alignments)


def save_alignments(target_path: str, id: str, alignments: List[TimedSegment]):
    """
    Saves a list of alignments to a csv file in the target location.

    Parameters
    ----------
    target_path: str
        The path to the folder to save the aligned files to.
    id: str
        The id of the file to save.
    alignments: List[TimedSegment]
        The alignments to save.

    """
    with open(os.path.join(target_path, id + ".csv"), "w+") as f:
        f.write("start,end,text\n")

        for segment in alignments:
            # Skip placeholder text
            if segment.text == placeholder_text:
                continue
            f.write(f"{segment.start},{segment.end},{segment.text}\n")
