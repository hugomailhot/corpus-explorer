"""Load data, process it, then define and populate the app layout.

"""
import argparse

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from gensim.matutils import corpus2csc
from gensim.models import LdaModel

import utils.dash_reusable_components as drc
import utils.figures as figs
import utils.nlp as nlp


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
data.text = data.text.map(nlp.normalize_text)

print('Building docterm matrix')
docterm, dictionary = nlp.get_docterm_matrix(data.text)
doclength = np.array([sum(x[1] for x in doc) for doc in docterm])

print('Training LDA model')
lda = LdaModel(docterm, num_topics=3)

print('Getting document topics')
doctopics = corpus2csc([lda.get_document_topics(doc) for doc in docterm])
termtopics = lda.get_topics()

print('Computing topic coordinates')
topic_coordinates = nlp.get_topic_coordinates(termtopics)
topic_proportions = nlp.get_topic_proportions(doctopics, doclength)

print('Computing term ranks per topic')
term_ranks = nlp.get_topic_term_ranks(docterm, termtopics)

# TODO: add term ranking display to the app

print('Building and populating app layout')
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[

        html.Div(
            id='div-top-controls',
            children=[
                drc.NamedSlider(
                    name='Topic Size Scaling',
                    id='slider-topic-size-scaling',
                    min=1,
                    max=2,
                    step=0.1,
                    value=1,
                ),
            ]
        ),

        html.Div(
            id='div-graphs',
        ),
    ],
)



@app.callback(
    Output('div-graphs', 'children'),
    [Input('slider-topic-size-scaling', 'value')])
def update_topic_scatter_plot_marker_sizes(topic_size_scaling):
    topic_size_scaling = float(topic_size_scaling)
    topic_scatter_plot = figs.serve_topic_scatter_plot(
        topic_coordinates, topic_proportions, topic_size_scaling,
    )

    return [
        html.Div(
            id='graph-topic-scatter-plot-container',
            children=dcc.Loading(
                className='graph-wrapper',
                children=dcc.Graph(
                    id='graph-topic-scatter-plot', figure=topic_scatter_plot
                ),
            ),
        ),
    ]

if __name__ == '__main__':
    app.run_server(debug=False)

