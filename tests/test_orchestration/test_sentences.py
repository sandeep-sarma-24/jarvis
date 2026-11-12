from app.orchestration.sentences import SentenceSplitter


def run(deltas, **kw):
    s = SentenceSplitter(**kw)
    out = []
    for d in deltas:
        out.extend(s.feed(d))
    tail = s.flush()
    return out, tail


def chars(text):
    return list(text)


def test_emits_sentence_as_soon_as_followed_by_space():
    s = SentenceSplitter()
    assert s.feed("The weather is nice today.") == []  # could still be "today.5"? no: needs whitespace
    assert s.feed(" And") == ["The weather is nice today."]
    assert s.flush() == "And"


def test_char_by_char_matches_whole():
    text = "Hello there, friend. The value is 3.5 percent! Is that right? Yes it is."
    whole, tail_whole = run([text])
    split, tail_split = run(chars(text))
    assert whole == split
    assert tail_whole == tail_split
    assert split == ["Hello there, friend.", "The value is 3.5 percent!", "Is that right?"]
    assert tail_split == "Yes it is."


def test_no_split_on_decimal_or_unfinished_number():
    s = SentenceSplitter()
    assert s.feed("It costs about 3.") == []
    assert s.feed("5 dollars in total. Next") == ["It costs about 3.5 dollars in total."]


def test_no_split_on_abbreviations():
    out, tail = run(["Ask Dr. Smith about it, e.g. tomorrow morning. Mr. Jones agrees. OK"])
    assert out == ["Ask Dr. Smith about it, e.g. tomorrow morning.", "Mr. Jones agrees."]
    assert tail == "OK"


def test_initials_are_not_sentence_ends():
    out, _ = run(["The book by J. R. Tolkien is long. Next"])
    assert out == ["The book by J. R. Tolkien is long."]


def test_short_fragments_coalesce_with_next_sentence():
    out, tail = run(["Sure. I can help with that. Ok"])
    assert out == ["Sure. I can help with that."]
    assert tail == "Ok"


def test_numbered_list_items_stay_with_their_text():
    out, _ = run(["1. Open the settings menu.\n2. Pick a voice.\n"])
    assert out == ["1. Open the settings menu.", "2. Pick a voice."]


def test_newline_is_a_boundary():
    out, tail = run(["First line without a period\nSecond line"])
    assert out == ["First line without a period"]
    assert tail == "Second line"


def test_long_first_sentence_emits_first_clause_early():
    s = SentenceSplitter(first_clause_chars=40)
    first = "When you consider all of the available options today, "
    assert s.feed(first) == ["When you consider all of the available options today,"]
    # only the first sentence is cut at a clause; later ones wait for a full stop
    assert s.feed("the best one is clear, and later, after the meeting, we decide") == []
    assert s.feed(". Done") == ["the best one is clear, and later, after the meeting, we decide."]


def test_short_first_sentence_not_cut_at_comma():
    s = SentenceSplitter(first_clause_chars=40)
    assert s.feed("Yes, of course, ") == []


def test_question_and_exclamation_with_closing_quote():
    out, _ = run(['He shouted "stop right now!" Then he left? ok'])
    assert out == ['He shouted "stop right now!"', "Then he left?"]


def test_flush_empty_returns_none():
    s = SentenceSplitter()
    assert s.flush() is None
    s.feed("   ")
    assert s.flush() is None


def test_ellipsis_is_a_boundary():
    out, _ = run(["Well, let me think about it... The answer is four. x"])
    assert out == ["Well, let me think about it...", "The answer is four."]
