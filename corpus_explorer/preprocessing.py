"""This is where we put all of the code to preprocess and shape a text collection
so that it can be used as input to a topic model learner.
"""

import re
from typing import Iterable


PUNCT_RE = r'[!"#$%&\'()*+,./:;<=>?@\^_`{|}~]'


def normalize_text(text: str) -> str:
    """Take raw text and apply several normalization steps to it.

    Specifically we perform:
        - lowercasing
        - numbers removal
        - punctuation removal

    Notes
    -----
    This function is currently just a minimal example. We might want to consider
    other normalization steps, such as:
        - stopword removal
        - lemmatization
        - stemming

    Parameters
    ----------
    text:
        Input text in its raw form.

    Returns
    -------
    Normalized text.

    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(PUNCT_RE, '', text)
    text = re.sub(r'\s\s+', ' ', text)  # Handle excess whitespace
    text = text.strip()  # No whitespace at start and end of string

    return text
