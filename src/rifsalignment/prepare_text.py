"""

"""

from typing import List
from num2words import num2words

import regex as re

def prepare_text(transcripts: List[str]):
    """
    Prepare the text for alignment.

    Parameters
    ----------
    transcripts: List[str]
        The transcripts to prepare.

    Returns
    -------
    List[str]
        The prepared transcripts.
    """

    # Strip every line from \n
    transcripts = [l.strip() for l in transcripts]

    # Join every line with space
    transcripts = ' '.join(transcripts)

    # Remove special characters
    transcripts = re.sub('[\,\?\!\-\;\:"]', '', transcripts)

    # Convert numbers to spoken equivelant
    transcripts = ''.join([num2words(w, lang='dk', to='year') if w.isdigit() else w for w in transcripts])

    # Split and uppercase sentences
    transcripts = [l.strip().upper() for l in transcripts.split('.')]

    # Add placeholder text in beginning
    transcripts.inser(0, "DETTE ER BARE EN PLADSHOLDER")

    return transcripts