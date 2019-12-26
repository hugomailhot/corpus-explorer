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
parser.add_argument(
    'n_topics',
    help='Number of topics in the LDA model',
    type=int,
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
lda = LdaModel(docterm, num_topics=args.n_topics)

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
                html.Div(
                    id='slider-term-relevance-lambda-container',
                    children=drc.NamedSlider(
                        name='Lambda',
                        id='slider-term-relevance-lambda',
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.6,
                    ),
                ),
                html.Div(
                    id='input-topic-id-container',
                    children=drc.NamedInput(
                        name='Topic ID',
                        id='input-topic-id',
                        type='number',
                        value=0,
                        min=0,
                        max=args.n_topics-1,
                        step=1,
                    ),
                ),
            ],
        ),

        html.Div(
            id='div-graphs',
            children=[
                html.Div(id='graph-topic-scatter-plot-container',
                         children=dcc.Graph(id='graph-topic-scatter-plot',
                                            style={'height': '85vh', 'width': '100%'})
                ),
                html.Div(id='graph-term-relevance-bar-plot-container'),
            ],
        ),
    ],
)


@app.callback(
    [Output('input-topic-id', 'value'),
     Output('graph-topic-scatter-plot-container', 'children')],
    [Input('graph-topic-scatter-plot', 'clickData')],
)
def update_topic_scatter_plot_selection(clickData):
    # Default to 0 during app initialization
    if clickData is None:
        topic_id = 0
    else:
        topic_id = clickData['points'][0]['customdata']['topic_id']

    topic_scatter_plot = figs.serve_topic_scatter_plot(
        topic_coordinates, topic_proportions, topic_id,
    )

    return [
        topic_id,
        dcc.Graph(
            id='graph-topic-scatter-plot',
            figure=topic_scatter_plot,
            style={'height': '85vh', 'width': '100%'},
        )
    ]


@app.callback(
    Output('graph-term-relevance-bar-plot-container', 'children'),
    [Input('slider-term-relevance-lambda', 'value'),
     Input('input-topic-id', 'value')])
def update_term_relevance_bar_plot(lambda_value, topic_id):
    term_relevance_bar_plot = figs.serve_term_relevance_bar_plot(
        term_ranks,
        dictionary,
        topic_id,
        lambda_value,
    )

    return dcc.Graph(
        id='graph-term-relevance-bar-plot',
        figure=term_relevance_bar_plot,
        style={'height': '85vh', 'widht': '100%'},
    )


if __name__ == '__main__':
    app.run_server(debug=True)
