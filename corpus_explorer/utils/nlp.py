"""This is where we put all of the code to preprocess and shape a text collection
so that it can be used as input to a topic model learner.
"""

import re
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing import cpu_count
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.matutils import corpus2csc
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from stop_words import get_stop_words


# Reasonable range for object sizes in Plotly
MIN_MARKER_SIZE = 20
MAX_MARKER_SIZE = 100

# Regex for punctuation in English text
PUNCT_RE = r'[!"#$%&\'()*+,./:;<=>?@\^_`{|}~]'

tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

############################
### STOPWORDS DEFINITION ###
############################
stopwords = get_stop_words('english')
stopwords += get_stop_words('spanish')
extra_stopwords = [
    # these aren't in the default set, but we should still filter them
    'said',
    'will',
    'one',
    'two',
    'three',
]
punctuation = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.', '/',
    ':', ';', '<', '=', '>', '?', '@', '^', '_', '`', '{', '|', '}', '~',
    'amp',
]
stopwords += extra_stopwords + punctuation
stopwords = set(stopwords)


# Check if NLTK's word tokenizer is installed
try:
    _ = word_tokenize('some sentence')
except LookupError:
    print('Downloading missing NLTK word tokenizer')
    import nltk
    nltk.download('punkt')

def tokenize_corpus(doc):
    return doc.lower().split()

def doc2bow(doc, dictionary):
    return dictionary.doc2bow(doc)

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

    with Pool(cpu_count() - 1) as p:

        tokenized_corpus = p.map(
            tokenize_corpus,
            corpus,
        )
        # tokenized_corpus = [doc.lower().split() for doc in corpus]
        dictionary = corpora.Dictionary(tokenized_corpus)

        docterm = p.starmap(
            doc2bow,
            ((doc, dictionary) for doc in tokenized_corpus),
        )

    return docterm, dictionary


def get_term_frequencies(docterm, termtopics, topic_proportions, doclength):
    """Compute overall term frequencies and estimated per-topic term frequencies.

    Parameters
    ----------
    docterm:
        asdf
    termtopics:
        asdf

    Returns
    -------
    Dict containing term frequencies for 'all' topics and for each topic id.

    """
    term_frequencies = {'all': defaultdict(int)}

    # Compute overall term frequencies
    for doc in docterm:
        for term_id, count in doc:
            term_frequencies['all'][term_id] += count

    # Estimate per-topic term frequencies
    n_terms = sum(doclength)
    for topic_id in range(termtopics.shape[0]):
        term_frequencies[topic_id] = {}
        for term_id in range(termtopics.shape[1]):
            # < 1% of the time, the estimate is larger than the total count.
            # We take the min to ensure per-topic frequency doesn't exceed
            # total frequency.
            term_frequencies[topic_id][term_id] = min(
                n_terms * topic_proportions[topic_id] * termtopics[topic_id][term_id],
                term_frequencies['all'][term_id],
            )

    return term_frequencies


def get_topic_coordinates(topicterms, method='pca'):
    """Compute a 2-dimensional embeddings of topics that reflects their
    distance from one another.

    Distance between two topics is defined here as the Jensen-Shannon divergence
    between the probability distributions over terms in the two topics.

    Parameters
    ----------
    topicterms: numpy.ndarray
        Matrix, |topics| x |terms|. Each row contains the term distribution
        for one topic.
    method: str
        Method used to obtain the 2-dimensional embeddings. Acceptable value
        are "pca", "mds" and "tsne".

    Returns
    -------
    numpy.array, |topics| x 2
    The ith row contains the x and y coordinates of the ith topic.

    """
    if method not in {'mds', 'pca', 'tsne'}:
        raise ValueError('method argument must be either "pca" or "tsne"')

    n_topics = topicterms.shape[0]
    distance_mat = np.zeros(shape=(n_topics, n_topics))

    # Start with upper triangular distance matrix.
    # The diagonal is already zeroes and is left untouched.
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            distance_mat[i][j] = jensenshannon(topicterms[i], topicterms[j])

    # Copy it on the lower half, flipping along the diagonal.
    distance_mat = distance_mat + distance_mat.T

    # Reduce dimensionality to obtain 2D topic embeddings.
    if method == 'pca':
        dimensionality_reducer = PCA(n_components=2)
    elif method == 'mds':
        dimensionality_reducer = MDS(n_components=2)
    else:
        dimensionality_reducer = TSNE(n_components=2)

    topic_coordinates = dimensionality_reducer.fit_transform(distance_mat)

    return topic_coordinates


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
    numpy.ndarray
        Vector, |topics|. Each value is the aggregate proportion of a topic over
        the entire set of documents.

    """
    len_weighted_doctopics = np.multiply(
        np.array(doctopics.todense()),  # todense give np.matrix, need array
        doclengths,
    )

    return np.sum(len_weighted_doctopics, axis=1) / np.sum(len_weighted_doctopics)


def get_topic_volume_over_time(df, doctopics, n_periods=20):
    """Compute the volume of documents attributable to each topic over n_periods
    consecutive time periods.

    Notes
    -----
    This makes use of a `timestamp` column, assumed to be present in the input
    dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Original dataset.
    doctopics: scipy.sparse.csc_matrix
        Topic distribution per document.
    n_periods: int
        Number of periods in which to slice the dataset.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, n_periods rows and n_topics columns. Each column
        holds a topic volume timeseries across all periods.

    """
    periods = pd.date_range(
        start=df.timestamp.min(),
        end=df.timestamp.max(),
        freq='M',
    )

    volume_over_time_df = pd.DataFrame(
        data=None,
        index=periods[:-1],
        columns=list(range(doctopics.shape[0])),
    )
    for i in range(len(periods) - 1):
        period_mask = (df.timestamp > periods[i]) & (df.timestamp <= periods[i+1])

        # works because idx is 0-indexed sequence of ints
        # TODO: find something that doesn't assume the df index
        period_idx = df[period_mask].index
        period_doctopics = doctopics[:, period_idx]
        topic_volumes = period_doctopics.sum(axis=1).squeeze()

        volume_over_time_df.loc[periods[i]] = topic_volumes

    return volume_over_time_df


def get_topic_term_ranks(
    docterm: List[Tuple[int]],
    termtopic: np.ndarray,
) -> Dict[float, Dict[int, List[int]]]:
    """Compute term relevance rankings for all topics.

    Notes
    -----
    The term relevance computation is described in this paper:
    https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

    Term relevance is defined as an interpolation between log(P(term|topic)) and
    log(P(term|topic) / P(term)). The interpolation is parameterized with a lambda
    term between 0 and 1. We compute:

    relevance(word, topic | lambda) =
        lambda * log(P(term|topic)) + (1 - lambda) * log(lift(term, topic))

    where lift(term, topic) = P(term|topic) / P(term).

    This function computes the rankings for all lambdas in the range [0, 1], with
    step size 0.1.

    Parameters
    ----------
    docterm
        A sparse document-term matrix in integer list repesentation.

    termtopic
        |topics| x |terms| np.ndarray
        Rows are term probabilities for a single topic.

    Returns
    -------
    A dict with the following structure:
        {
            lambda_value_1: {
                topic_1: (term_id_1, term_id_2, term_id_3, ...),
                topic_2: (term_id_x, term_id_y, term_id_z, ...),
                ...
            },
            lambda_value_2: ...,
            ...
        }

    """
    # Convert docterm into scipy matrix and compute P(term) for all terms
    docterm_mat = corpus2csc(docterm)
    p_term = np.asarray(docterm_mat.sum(axis=1) / docterm_mat.sum()).squeeze()

    term_ranks = {}

    for lam in np.arange(0, 1.1, step=0.1):
        lam = round(lam, 1)  # round to avoid floating point imprecision stuff
        term_ranks[lam] = {}

        for topic_id in range(termtopic.shape[0]):
            p_term_topic = termtopic[topic_id]
            lift = np.divide(p_term_topic, p_term)
            term_topic_relevance = (
                lam * np.log(p_term_topic) +
                (1 - lam) * np.log(lift)
            )
            # Rank term ids in decreasing order of relevance
            ranked_term_ids = np.argsort(term_topic_relevance)[::-1].tolist()
            term_ranks[lam][topic_id] = ranked_term_ids

    return term_ranks


def normalize_text(text: str) -> str:
    """Take raw text and apply several normalization steps to it.

    Specifically we perform:
        - lowercasing
        - numbers removal
        - punctuation removal
        - stopword removal

    Notes
    -----
    This function is currently just a minimal example. We might want to consider
    other normalization steps, such as:
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
    # text = re.sub(r'\d+', '', text)
    # text = re.sub(PUNCT_RE, '', text)
    # text = re.sub(r'\s\s+', ' ', text)  # Handle excess whitespace
    text = text.strip()  # No whitespace at start and end of string

    stopwords = get_stop_words('english')
    extra_stopwords = [
        # these aren't in the default set, but we should still filter them
        'said',
        'will',
        'one',
        'two',
        'three',
    ]
    stopwords += extra_stopwords
    text = ' '.join(x for x in word_tokenize(text) if x not in stopwords)

    return text


def normalize_tweet(text):
    """Take raw Tweet and apply several normalization steps to it.

    Specifically we perform:
        - lowercasing
        - numbers removal
        - punctuation removal
        - stopword removal

    Notes
    -----
    This function is currently just a minimal example. We might want to consider
    other normalization steps, such as:
        - lemmatization
        - stemming

    Parameters
    ----------
    text:
        Input tweet in its raw form.

    Returns
    -------
    Normalized tweet.

    """
    text = ' '.join(x for x in tweet_tokenizer.tokenize(text) if x not in stopwords)

    return text


def papply(func, df, *args, max_cores=None):
    """Apply function over the DataFrame using parallel processes.

    max_cores is there in case leveraging all cores would results
    in memory shortage.

    Arguments
    ---------
    func:
        a function with arguments(df, *args), must return a df
    df:
    args:
        optional additional arguments for the function
    max_cores: optional int
        If set, will restrict how many cores the work is split over

    Returns
    -------
        The df after having applied func to all its rows.

    """
    cores = cpu_count() - 1

    if max_cores:
        cores = min(cores, max_cores)

    chunks = np.array_split(df, cores)

    with Pool(cores) as p:
        if args:
            arg_tuples = ([c] + list(args) for c in chunks)
            result = pd.concat(p.starmap(func, arg_tuples))
        else:
            result = pd.concat(p.map(func, chunks))

    return result


def parallel_normalize_tweets(tweets):

    n_cores = cpu_count() - 1

    with Pool(n_cores) as p:
        res = p.map(normalize_tweet, tweets)

    return res

