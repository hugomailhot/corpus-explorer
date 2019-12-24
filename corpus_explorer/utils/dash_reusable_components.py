"""Useful abstractions that handle repetitive boilerplate attribute setting.

"""
import dash_core_components as dcc
import dash_html_components as html


def NamedSlider(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**kwargs)),
        ],
    )
