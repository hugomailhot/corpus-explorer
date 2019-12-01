"""This script put everything together to go from a well-formed corpus
to an HTML visualization of the topics in the corpus.

"""


import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gensim.matutils import corpus2csc
from gensim.models import LdaModel
from sklearn.preprocessing import MinMaxScaler

from corpus_explorer.utils import normalize_text
from corpus_explorer.utils import get_docterm_matrix
from corpus_explorer.utils import get_topic_coordinates
from corpus_explorer.utils import get_topic_proportions


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Learn topic model and generate visualization data.',
    )
    parser.add_argument(
        'input_filepath',
        help='Filepath of the input corpus',
        type=str,
    )
    args = parser.parse_args()
    data = pd.read_parquet(args.input_filepath)

    data.text = data.text.map(normalize_text)

    docterm, dictionary = get_docterm_matrix(data.text)
    doclength = np.array([sum(x[1] for x in doc) for doc in docterm])

    lda = LdaModel(docterm, num_topics=3)
    doctopics = corpus2csc([lda.get_document_topics(doc) for doc in docterm])
    termtopics = lda.get_topics()

    topic_coordinates = get_topic_coordinates(termtopics)
    topics_x_coords = topic_coordinates[:, 0]
    topics_y_coords = topic_coordinates[:, 1]

    topic_proportions = get_topic_proportions(doctopics, doclength)
    # Scale proportion values to adequate Plotly marker size values
    scaler = MinMaxScaler(feature_range=(20, 100))
    topic_sizes = scaler.fit_transform(topic_proportions.reshape(-1, 1))

    fig = go.Figure(
        go.Scatter(
            x=topics_x_coords,
            y=topics_y_coords,
            mode='markers',
            marker_size=topic_sizes,
        ),
    )

    fig.write_html('test.html')

