"""This is where we put all of the code to preprocess and shape a text collection
so that it can be used as input to a topic model learner.
"""

import re
from typing import Iterable
from typing import List
from typing import Tuple

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input
from dash.dependencies import Output
from gensim import corpora
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


# Reasonable range for object sizes in Plotly
MIN_MARKER_SIZE = 20
MAX_MARKER_SIZE = 100

# Regex for punctuation in English text
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
        are "pca" and "tsne".

    Returns
    -------
    numpy.array, |topics| x 2
    The ith row contains the x and y coordinates of the ith topic.

    """
    if method not in {'pca', 'tsne'}:
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


def generate_topic_scatter_plot(topic_coordinates, topic_proportions):
    """With 2-D topic embeddings and their proportion across a corpus, generate
    a scatter plot that represents their relative distance and sizes.

    Parameters
    ----------
    topic_coordinates: numpy.array, |topics| x 2
        2-dimensional coordinates representing topic positions.
    topic_propoertions: numpy.ndarray, |topics|
        Vector of topic proportion in a corpus.

    Returns
    -------
    plotly.graph_objects.Figure
        Graph object, ready to be displayed or written to disk.

    """
    x_coords = topic_coordinates[:, 0]
    y_coords = topic_coordinates[:, 1]

    # Scale proportion values to adequate Plotly marker size values
    scaler = MinMaxScaler(feature_range=(MIN_MARKER_SIZE, MAX_MARKER_SIZE))
    topic_sizes = scaler.fit_transform(topic_proportions.reshape(-1, 1))

    return go.Figure(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker_size=topic_sizes,
        ),
    )


def get_topic_term_ranks(
    docterm: List[Tuple[int]],
    topicterm: np.ndarray
) -> Dict[float, Dict[int, List[Tuple[str, float, float]]]]:
    """Compute term relevance rankings for all topics.

    Notes
    -----
    The term relevance computation is described in this paper:
    https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

    Term relevance is defined as an interpolation between log(P(term|topic)) and
    log(P(term|topic) / P(term)). The interpolation is parameterized with a lambda
    term between 0 and 1. We compute:

    relevance(word, topic | lambda) =
        lambda * log(P(term|topic)) + (1 - lambda) * log(P(term|topic) / P(term))

    This function computes the rankings for all lambdas in the range [0, 1], with
    step size 0.1.

    Parameters
    ----------
    docterm:
        A sparse document-term matrix in integer list repesentation.

    topicterm
        |topics| x |terms| np.ndarray
        Rows are term probabilities for a single topic.

    Returns
    -------
    A dict with the following structure:
        {
            lambda_value_1: {
                topic_1: (term1, term2, term3, ...),
                topic_2: (termx, termy, termz, ...),
                ...
            },
            lambda_value_2: ...,
            ...
        }

    """
    # Convert docterm into scipy matrix and compute P(term)
    dt = corpus2csc(docterm)

    term_ranks = {}

    return term_ranks


def generate_visualization(topic_coordinates, topic_proportions):

    x_coords = topic_coordinates[:, 0]
    y_coords = topic_coordinates[:, 1]

    # Scale proportion values to adequate Plotly marker size values
    scaler = MinMaxScaler(feature_range=(MIN_MARKER_SIZE, MAX_MARKER_SIZE))
    topic_sizes = scaler.fit_transform(topic_proportions.reshape(-1, 1))

    app = dash.Dash()

    slider = dcc.Slider(
        id='slider_input',
        min=1,
        max=2,
        step=0.1,
        value=1,
    )

    graph = dcc.Graph(id='example')
    graph_layout = go.Layout(
        title='Plot',
        yaxis=dict(range=[-0.5, 0.5]),
        xaxis=dict(range=[-0.5, 0.5]),
    )

    app.layout = html.Div(
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Level"),
                        slider,
                    ],
                    style={'width': 200},
                ),
                html.Div(style={'height': 10, 'width': 200}),
                graph,
            ],
            style={'width': 500, 'height': 200, 'display': 'inline-block'},
        ),
    )

    # callback - 1 (from slider)
    @app.callback(Output('example', 'figure'),
                  [Input('slider_input', 'value')])
    def update_plot(slider_input):
        q = float(slider_input)

        figure = {
            'data': [
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    marker_size=topic_sizes * q,
                ),
            ],
            'layout': graph_layout,
        }

        return figure

    return app
