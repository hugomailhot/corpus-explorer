"""This script put everything together to go from a well-formed corpus
to an HTML visualization of the topics in the corpus.

"""


import argparse

import numpy as np
import pandas as pd
from gensim.matutils import corpus2csc
from gensim.models import LdaModel

from corpus_explorer.utils import generate_visualization
from corpus_explorer.utils import get_docterm_matrix
from corpus_explorer.utils import get_topic_coordinates
from corpus_explorer.utils import get_topic_proportions
from corpus_explorer.utils import get_topic_term_ranks
from corpus_explorer.utils import normalize_text


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

    print('Reading dataset')
    data = pd.read_parquet(args.input_filepath)

    print('Normalizing text')
    data.text = data.text.map(normalize_text)

    print('Building docterm matrix')
    docterm, dictionary = get_docterm_matrix(data.text)
    doclength = np.array([sum(x[1] for x in doc) for doc in docterm])

    print('Training LDA model')
    lda = LdaModel(docterm, num_topics=40)

    print('Getting document topics')
    doctopics = corpus2csc([lda.get_document_topics(doc) for doc in docterm])
    termtopics = lda.get_topics()

    print('Computing topic coordinates')
    topic_coordinates = get_topic_coordinates(termtopics)
    topic_proportions = get_topic_proportions(doctopics, doclength)

    print('Computing term ranks per topic')
    term_ranks = get_topic_term_ranks(docterm, termtopics)

    # TODO: add term ranking display to the app

    print('Launching app')
    app = generate_visualization(topic_coordinates, topic_proportions)
    app.run_server(debug=False)
