"""
Main way of interacting with this package.
"""

from rifsalignment.algorithms.base_alignment import BaseAlignment

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
        The path to the folder to save the aligned files to. If None, the files will be saved to the same folder as the csv file.
    verbose: bool
        Whether to print the alignments progress with steps.
    quiet: bool
        Prints nothing.
    """

    # Load csv file
    all_csv = pd.read_csv(os.path.join(data_path, "all.csv"))

    # Retrieve audio and text files and parse to align
    all_alignments = {}
    for i, row in all_csv.iterrows():
        if verbose and not quiet:
            print(f"Aligning {row['id']}")
        alignments = align_method.align(
            audio_file=os.path.join(data_path, "audio", row["id"] + ".wav"),
            text_file=os.path.join(data_path, "text", row["id"] + ".txt"),
            model=model,
        )
        all_alignments[row["id"]] = alignments

    # Save alignments to csv
    for id, alignments in all_alignments.items():

        if target_path is None:
            target_path = data_path
        if not os.path.exists(os.path.join(target_path, "alignments")):
            os.makedirs(os.path.join(target_path, "alignments"))

        with open(os.path.join(target_path, "alignments", id + ".csv"), "w") as f:
            f.write("start,end,text")

            for segment in alignments:
                f.write(f"{segment.start},{segment.end},{segment.text}")
