"""
Main way of interacting with this package.
"""

from rifsalignment.basealign import BaseAlignment


def align_csv(
    data_path: str,
    align_method: BaseAlignment,
    target_path: str = None,
    verbose: bool = False,
    quiet: bool = False,
):
    """

    Parameters
    ----------
    data_path: str
        The path to the csv file containing the audio and text files to align. Should of the rifs format.
    align_method: BaseAlignment
        The method to use for alignment. Can be one of the following: CTC
    target_path: str
        The path to the folder to save the aligned files to. If None, the files will be saved to the same folder as the csv file.
    verbose: bool
            Whether to print the alignments progress with steps.
    quiet: bool
        Prints nothing.
    """
    print("Not implemented yet")
    pass
    # Load csv file

    # Retrieve audio and text files and parse to align
