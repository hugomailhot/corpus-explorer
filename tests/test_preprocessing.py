from corpus_explorer.preprocessing import normalize_text


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
