"""Utility functions to draw figures for the app.
Every function here returns a go.Figure object.

"""
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler


def serve_term_relevance_bar_plot(term_ranks, term_frequencies, dictionary, topic_id, lam):
    N = 30
    top_term_ids = term_ranks[lam][topic_id][:N][::-1]
    top_terms = [dictionary[term_id] for term_id in top_term_ids]
    top_terms_total_freqs = [term_frequencies['all'][x] for x in top_term_ids]
    top_terms_topic_freqs = [term_frequencies[topic_id][x] for x in top_term_ids]

    data = [
        go.Bar(
            x=top_terms_total_freqs,
            y=top_terms,
            orientation='h',
            name='Overall term frequency',
        ),
        go.Bar(
            x=top_terms_topic_freqs,
            y=top_terms,
            orientation='h',
            name='Estimated term frequency within the selected topic',
        ),
    ]

    layout = go.Layout(
        title=f'{N} most relevant topic terms',
        barmode='overlay',
        legend=dict(x=0, y=-0.1),
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
    scaler = MinMaxScaler(feature_range=(40, 200))
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
        xaxis={
            'range': [min(x_coords) - x_pad, max(x_coords) + x_pad],
        },
        yaxis={
            'range': [min(y_coords) - y_pad, max(y_coords) + y_pad],
        },
        transition={'duration': 200},  # animate from previous plot to next
        # plot_bgcolor="#DDDDDB",
        # paper_bgcolor="#282b38",
        # font={"color": "#a5b1cd"},
    )

    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_topic_volume_over_time_plot(
    volume_over_time_df, selected_topic,
):

    # marker = {
    #    'fillcolor': ["rgba(25, 25, 25, 0.5)" for x in range(volume_over_time_df.shape[1])]
    # }
    # if selected_topic is not None:
    #     marker['opacity'][selected_topic] = 1

    data = []
    for topic_id in volume_over_time_df.columns:
        data.append(
            go.Scatter(
                x=volume_over_time_df.index,
                y=volume_over_time_df[topic_id],
                customdata=[
                    {'topic_id': topic_id}
                    for x in range(volume_over_time_df.shape[0])
                ],
                hoveron='fills+points',
                # marker=marker,
                mode='lines',
                stackgroup='A',
                fillcolor=(
                    'rgba(50, 50, 50, 1)'
                    if topic_id == selected_topic
                    else 'rgba(50, 50, 50, 0.2)'
                ),
            ),
        )

    figure = go.Figure(data=data)

    return figure
