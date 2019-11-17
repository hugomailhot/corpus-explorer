import numpy as np

from corpus_explorer.utils import normalize_text, get_topic_proportions


def test_normalize_text_handles_excess_whitespace():
    raw_text = 'why the long                            pause'
    expected = 'why the long pause'
    assert normalize_text(raw_text) == expected

    raw_text = '''
    now for something
    different

    '''
    expected = 'now for something different'
    assert normalize_text(raw_text) == expected


def test_normalize_text_removes_numbers():
    raw_text = 'ya hate 2 see it'
    expected = 'ya hate see it'
    assert normalize_text(raw_text) == expected


def test_normalize_text_removes_punctuation():
    raw_text = 'that guy: he really takes the "the" out of "psychotherapist", amirite?'
    expected = 'that guy he really takes the the out of psychotherapist amirite'
    assert normalize_text(raw_text) == expected


def test_get_topic_proportions_return_correction_proportions():
    doctopics = np.array([
        [0.2, 0.3, 0.5],
        [0.4, 0.2, 0.4],
    ])
    doclengths = np.array([2, 3])

    expected = np.array([0.32, 0.24, 0.44])
    actual = get_topic_proportions(doctopics, doclengths)

    # Using allclose instead of array_equal here, since floating point
    # imprecision causes inequality with a difference of 5e-17
    assert np.allclose(actual, expected)
