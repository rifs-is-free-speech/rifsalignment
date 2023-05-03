"""
Preprocess the transcripts for alignment.
"""

import regex as re

from typing import List
from num2words import num2words

from rifsalignment.datamodels import placeholder_text


def prepare_text(transcripts: List[str], prepend_placeholder: bool = False):
    """
    Prepare the text for alignment.

    Parameters
    ----------
    transcripts: List[str]
        The transcripts to prepare.
    prepend_placeholder: bool
        Whether to prepend a placeholder text in the beginning of the transcripts.
        Some methods, like CTC, benefit from this. Default is False.

    Returns
    -------
    List[str]
        The prepared transcripts.
    """


    # Strip every line from \n
    transcripts = [line.strip() for line in transcripts]

    # Add period at end of every line if not present
    transcripts = [line + "." if not line.endswith(".") else line for line in transcripts]

    # Join every line with space
    transcripts = " ".join(transcripts)

    # Remove special characters
    transcripts = re.sub('[\,\?\!\-\;\:"]', "", transcripts)  # noqa: W605

    # Convert numbers to spoken equivelant
    transcripts = " ".join(
        [num2words(w, lang="dk", to="year") if w.isdigit() else w for w in transcripts.split(" ")]
    )

    # Split and uppercase sentences. Up to -1 to remove last empty sentence after last period.
    transcripts = [line.strip().lower() for line in transcripts.split(".")[:-1]]

    # Add placeholder text in beginning
    if prepend_placeholder:
        transcripts.insert(0, placeholder_text)

    return transcripts
