"""

"""

from typing import List
from num2words import num2words

import regex as re


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
    transcripts = [l.strip() for l in transcripts]

    # Join every line with space
    transcripts = " ".join(transcripts)

    # Remove special characters
    transcripts = re.sub('[\,\?\!\-\;\:"]', "", transcripts)

    # Convert numbers to spoken equivelant
    transcripts = "".join(
        [num2words(w, lang="dk", to="year") if w.isdigit() else w for w in transcripts]
    )

    # Split and uppercase sentences
    transcripts = [l.strip().upper() for l in transcripts.split(".")]

    # Add placeholder text in beginning
    if prepend_placeholder:
        transcripts.insert(0, "DETTE ER BARE EN PLADSHOLDER")

    return transcripts
