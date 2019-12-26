"""Utility functions to draw figures for the app.
Every function here returns a go.Figure object.

"""
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler


def serve_term_relevance_bar_plot(term_ranks, dictionary, topic_id, lam):
    N = 30
    top_N_terms = [dictionary[term_id] for term_id in term_ranks[lam][topic_id][:N]]

    data = go.Bar(
        x=list(range(1, N + 1)),
        y=top_N_terms,
        orientation='h',
    )

    layout = go.Layout(
        title=f'{N} most relevant topic terms',
        # plot_bgcolor="#282b38",
        # paper_bgcolor="#282b38",
        # font={"color": "#a5b1cd"},
    )

    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_topic_scatter_plot(
    topic_coordinates, topic_proportions, selected_topic,
):

    marker = {'color': ['#80E1EA' for x in range(len(topic_proportions))]}
    if selected_topic is not None:
        marker['color'][selected_topic] = '#E98364'

    x_coords = topic_coordinates[:, 0]
    y_coords = topic_coordinates[:, 1]

    # Scale proportion values to adequate Plotly marker size values
    scaler = MinMaxScaler(feature_range=(20, 100))
    topic_sizes = scaler.fit_transform(topic_proportions.reshape(-1, 1))

    data = go.Scatter(
        x=x_coords,
        y=y_coords,
        customdata=[{'topic_id': x} for x in range(len(x_coords))],
        mode='markers+text',
        marker_size=topic_sizes,
        marker=marker,
        text=[str(x) for x in range(len(x_coords))],
        textposition='middle center',
        textfont={'family': 'Arial, sans-serif', 'color': 'white', 'size': 21},

    )
    x_pad = 0.05
    y_pad = 0.05
    layout = go.Layout(
        title='Inter-topic distance',
        xaxis=dict(range=[min(x_coords) - x_pad, max(x_coords) + x_pad]),
        yaxis=dict(range=[min(y_coords) - y_pad, max(y_coords) + y_pad]),
        transition={'duration': 500},  # animate from previous plot to next
        # plot_bgcolor="#282b38",
        # paper_bgcolor="#282b38",
        # font={"color": "#a5b1cd"},
    )

    figure = go.Figure(data=data, layout=layout)

    return figure
