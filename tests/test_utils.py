import numpy as np
import pytest
import scipy

from corpus_explorer.utils.nlp import normalize_text
from corpus_explorer.utils.nlp import get_topic_proportions
from corpus_explorer.utils.nlp import get_topic_coordinates

def test_normalize_text_handles_excess_whitespace():
    raw_text = 'why the long                            pause'
    expected = 'long pause'
    assert normalize_text(raw_text) == expected

    raw_text = '''
    now for something
    different

    '''
    expected = 'now something different'
    assert normalize_text(raw_text) == expected


def test_normalize_text_removes_numbers():
    raw_text = 'ya hate 2 see it'
    expected = 'ya hate see'
    assert normalize_text(raw_text) == expected


def test_normalize_text_removes_punctuation():
    raw_text = 'that guy: he really takes the "the" out of "psychotherapist", amirite?'
    expected = 'guy really takes psychotherapist amirite'
    assert normalize_text(raw_text) == expected


def test_get_topic_proportions_return_correction_proportions():
    doctopics = scipy.sparse.csc_matrix(
        np.array([
            [0.2, 0.4],
            [0.3, 0.2],
            [0.5, 0.4],
        ]),
    )
    doclengths = np.array([2, 3])

    expected = np.array([0.32, 0.24, 0.44])

    actual = get_topic_proportions(doctopics, doclengths)

    # Using allclose instead of array_equal here, since floating point
    # imprecision causes inequality with a difference of 5e-17
    assert np.allclose(actual, expected)

def test_get_topic_coordinates_returns_expected_shape():
    topicterms = np.array([
        [0.1, 0.2, 0.0, 0.0, 0.7],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.0, 0.7],
        [0.1, 0.0, 0.9, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ])

    xy_coords = get_topic_coordinates(topicterms)
    mat_shape = xy_coords.shape
    expected = (topicterms.shape[0], 2)
    assert mat_shape == expected


def test_get_topic_coordinates_raises_error_on_invalid_method_value():
    topicterms = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        get_topic_coordinates(topicterms, method='whoopsies')
