import numpy as np
import pandas as pd
from gensim.matutils import corpus2csc
from gensim.models import LdaModel

from corpus_explorer.preprocessing import normalize_text
from corpus_explorer.preprocessing import get_docterm_matrix


def get_lda_output(corpus: pd.DataFrame):
    """Given a text corpus, return all that's needed for the visualization
    component.

    Parameters
    ----------
    corpus:
        A pd.DataFrame with columns 'ID', 'text', 'timestamp', 'tags'.

    Returns
    -------
    complete this

    """
    corpus_text = corpus['text']
    corpus_meta = corpus.drop(columns='text')

    corpus_text = corpus_text.map(lambda x: normalize_text(x))
    docterm, dictionary = get_docterm_matrix(corpus_text)
    doclength = np.array([sum(x[1] for x in doc) for doc in docterm])

    lda = LdaModel(docterm, num_topics=3)
    doctopics = [lda.get_document_topics(doc) for doc in docterm]
    termtopics = lda.get_topics()

    return {
        'doctopics': corpus2csc(doctopics),
        'docterm': corpus2csc(docterm),
        'termtopics': termtopics,
        'dictionary': dictionary,
        'corpus_meta': corpus_meta,
        'doclength': doclength,
    }

