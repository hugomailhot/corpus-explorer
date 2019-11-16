"""This is where we put all of the code to preprocess and shape a text collection
so that it can be used as input to a topic model learner.
"""

import re
from typing import Iterable, List, Tuple

from gensim import corpora


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


def get_docterm_matrix(corpus: Iterable[str]) -> List[Tuple[int]]:
    """Turn a collection of texts into a document-term matrix.

    This function returns a list of tuples of ints, which is an acceptable
    value to the 'corpus' parameter in gensim's model classes.

    See here for example:
    https://radimrehurek.com/gensim/models/ldamulticore.html#gensim.models.ldamulticore.LdaMulticore

    Notes
    -----
    The current implementation forces using whitespace as a tokenization rule.
    If we feel like we need more control, we could add a 'tokenizer' param that
    accepts any tokenizing function that implements str -> List[str].

    Parameters
    ----------
    corpus:
        An iterable of strings. In this project this is likely a pd.Series of strings,
        but it doesn't have to be.

    Returns
    -------
    A sparse document-term matrix in integer list repesentation.

    """
    tokenized_corpus = [doc.split() for doc in corpus]
    dictionary = corpora.Dictionary(tokenized_corpus)
    docterm = [dictionary.doc2bow(doc) for doc in tokenized_corpus]

    return docterm, dictionary


def get_topic_proportions(doctopics, doclengths):
    """Compute the distribution of topics over a set of documents.

    Parameters
    ----------
    doctopics: scipy.sparse.csc_matrix
        Sparse matrix, |documents| x |topics|. Each row contains the topic
        distribution for one document.
    doclengths: np.array
        Vector, |documents|. Each value if the length of a single document.

    Returns
    -------
    Vector, |topics|. Each value is the aggregate proportion of a topic over
    the entire set of documents.

    """
    len_weighted_doctopics = doctopics * doclengths[:, None]

    return np.sum(len_weighted_doctopics, axis=0) / np.sum(len_weighted_doctopics)
