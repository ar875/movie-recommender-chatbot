"""
Unit tests for recommender.py

These are deliberately built around the REAL bugs found while building this
project (see the README's "Key Engineering Challenges" section) — each one
is a regression test, not a generic example. If any of these start failing,
it means a fix that was made during development has been accidentally
undone.

Tests don't load the real 300MB model file or hit the network: a lightweight
fake MovieRecommender instance is built with a small in-memory catalog and a
mocked ALS model, so the whole suite runs in well under a second and needs
no external dependencies.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from recommender import MovieRecommender, run_agent, TOOL_SCHEMAS, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Fixture: a small, fast, fully in-memory MovieRecommender
# ---------------------------------------------------------------------------

@pytest.fixture
def recommender():
    """Build a MovieRecommender instance without touching any real files.

    __new__ skips __init__ (which normally loads the CSV + pickle), so we
    can set up a small, deterministic catalog by hand instead.
    """
    rec = MovieRecommender.__new__(MovieRecommender)

    # A tiny catalog including the exact title-formatting edge cases that
    # caused real bugs during development: duplicate/near-duplicate titles,
    # and MovieLens's "Title, The (YYYY)" convention.
    rec.movies = pd.DataFrame([
        {"movieId": 1, "title": "Inception (2010)", "genres": "Action|Sci-Fi"},
        {"movieId": 2, "title": "Dark Knight, The (2008)", "genres": "Action|Crime"},
        {"movieId": 3, "title": "Dark Knight Rises, The (2012)", "genres": "Action|Crime"},
        {"movieId": 4, "title": "The Dark Knight (2011)", "genres": "Drama"},  # obscure duplicate/mislabeled entry
        {"movieId": 5, "title": "Toy Story (1995)", "genres": "Animation|Comedy"},
        {"movieId": 6, "title": "Forrest Gump (1994)", "genres": "Drama|Romance"},
        {"movieId": 7, "title": "Interstellar (2014)", "genres": "Adventure|Drama|Sci-Fi"},
    ])
    rec.movie_titles = rec.movies["title"].tolist()
    rec.movie_titles_clean = [
        rec._normalize_article(rec._strip_year(t)) for t in rec.movie_titles
    ]

    # Rating counts: movieId 2 (the real Dark Knight) is far more rated than
    # movieId 4 (the obscure duplicate) — this is what the popularity
    # tie-break relies on to pick the right one.
    rec.movie_rating_counts = pd.Series(
        {1: 50000, 2: 80000, 3: 60000, 4: 10, 5: 70000, 6: 65000, 7: 55000}
    )
    rec.popularity_ranking = rec.movie_rating_counts.sort_values(ascending=False).index.tolist()

    rec.user_to_idx = {1: 0, 2: 1}
    rec.movie_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    rec.idx_to_movie = {v: k for k, v in rec.movie_to_idx.items()}

    rec.model = MagicMock()
    rec.user_item_matrix = MagicMock()

    return rec


# ---------------------------------------------------------------------------
# find_movie_id — title matching
# ---------------------------------------------------------------------------

def test_find_movie_id_exact_match(recommender):
    movie_id, title = recommender.find_movie_id("Inception")
    assert movie_id == 1
    assert title == "Inception (2010)"


def test_find_movie_id_is_case_insensitive(recommender):
    movie_id, _ = recommender.find_movie_id("inception")
    assert movie_id == 1


def test_find_movie_id_handles_article_position_regression(recommender):
    """Regression test: 'The Dark Knight' must resolve to the REAL movie
    (movieId 2, the famous 2008 Nolan film with 80,000 ratings), not the
    obscure 2011 duplicate (movieId 4, only 10 ratings) — even though the
    obscure one is a literal exact-string match before article-normalization.

    This was a real, confirmed bug: without normalizing 'The X' vs 'X, The',
    find_movie_id matched the wrong, nearly-unrated catalog entry.
    """
    movie_id, title = recommender.find_movie_id("The Dark Knight")
    assert movie_id == 2
    assert title == "Dark Knight, The (2008)"


def test_find_movie_id_popularity_tiebreak(recommender):
    """When multiple catalog entries are equally valid matches, the one
    with more ratings should win (a proxy for 'the real movie' vs. a
    rarely-rated duplicate or mislabeled entry)."""
    movie_id, _ = recommender.find_movie_id("Dark Knight")
    # "Dark Knight" is a substring of both movieId 2 and movieId 3's
    # cleaned titles ("dark knight" and "dark knight rises") — only 2
    # matches exactly after cleaning, so this should resolve to it.
    assert movie_id == 2


def test_find_movie_id_fuzzy_typo(recommender):
    movie_id, title = recommender.find_movie_id("Inceptoin")  # typo
    assert movie_id == 1


def test_find_movie_id_strips_trailing_punctuation(recommender):
    movie_id, _ = recommender.find_movie_id("Toy Story!!!")
    assert movie_id == 5


def test_find_movie_id_no_match_returns_none(recommender):
    movie_id, title = recommender.find_movie_id("Completely Fictional Movie Title Xyz")
    assert movie_id is None
    assert title is None


# ---------------------------------------------------------------------------
# tool_lookup_movie_info
# ---------------------------------------------------------------------------

def test_tool_lookup_movie_info_found(recommender):
    result = recommender.tool_lookup_movie_info("Toy Story")
    assert result["method"] == "lookup"
    assert result["found"] is True
    assert result["matched_title"] == "Toy Story (1995)"
    assert "Animation" in result["genres"]


def test_tool_lookup_movie_info_not_found(recommender):
    result = recommender.tool_lookup_movie_info("Nonexistent Movie Zzz")
    assert result["method"] == "lookup_not_found"
    assert result["found"] is False


# ---------------------------------------------------------------------------
# tool_get_recommendations
# ---------------------------------------------------------------------------

def test_tool_get_recommendations_personalized_for_known_user(recommender):
    recommender.model.recommend.return_value = ([4, 5], [0.9, 0.8])
    # movieId 4 and 5 correspond to idx 3 and 4 in idx_to_movie
    recommender.model.recommend.return_value = ([3, 4], [0.9, 0.8])

    result = recommender.tool_get_recommendations(user_id=1, top_k=2)

    assert result["method"] == "personalized_als"
    assert "The Dark Knight (2011)" in result["recommendations"]
    assert "Toy Story (1995)" in result["recommendations"]


def test_tool_get_recommendations_item_similarity(recommender):
    # similar_items returns (ids, scores); id 1 is Dark Knight's own index (2),
    # which should be filtered out of the results since a movie isn't
    # "similar to itself" in the output.
    recommender.model.similar_items.return_value = ([1, 4, 5], [1.0, 0.8, 0.7])

    result = recommender.tool_get_recommendations(seed_movie_title="Inception", top_k=2)

    assert result["method"] == "item_similarity"
    assert result["seed_matched_to"] == "Inception (2010)"
    assert "Inception (2010)" not in result["recommendations"]  # must not recommend itself
    assert len(result["recommendations"]) == 2


def test_tool_get_recommendations_hybrid_reranks_by_genre_overlap(recommender):
    """This is the core test for the hybrid (ALS + genre) recommender.

    Pure ALS similarity ranks candidates purely by raw score: here that
    would put Forrest Gump (no genre overlap with Inception) ahead of
    Interstellar (shares Sci-Fi with Inception), since 0.70 > 0.65.

    The hybrid blend should flip this for the final top-2: Interstellar's
    genre overlap should be enough to outrank Forrest Gump's higher but
    thematically unrelated raw ALS score. Dark Knight (highest raw score,
    and some genre overlap via Action) should still be #1 either way.
    """
    dark_knight_idx = recommender.movie_to_idx[2]
    forrest_gump_idx = recommender.movie_to_idx[6]
    interstellar_idx = recommender.movie_to_idx[7]
    toy_story_idx = recommender.movie_to_idx[5]

    # Raw ALS order (by score alone) would be: Dark Knight, Forrest Gump,
    # Interstellar, Toy Story.
    recommender.model.similar_items.return_value = (
        [dark_knight_idx, forrest_gump_idx, interstellar_idx, toy_story_idx],
        [0.90, 0.70, 0.65, 0.50],
    )

    result = recommender.tool_get_recommendations(seed_movie_title="Inception", top_k=2)

    assert result["method"] == "item_similarity"
    # Interstellar (genre overlap) must outrank Forrest Gump (no overlap)
    # in the final top-2, even though Forrest Gump had the higher raw score.
    assert result["recommendations"] == ["Dark Knight, The (2008)", "Interstellar (2014)"]
    assert "Forrest Gump (1994)" not in result["recommendations"]

    # Also confirms the over-fetch pattern: the implementation should ask
    # the model for more than top_k candidates (fetch_k=20 for top_k=2)
    # so there's an actual pool to re-rank, not just top_k raw picks.
    _, call_kwargs = recommender.model.similar_items.call_args
    assert call_kwargs["N"] == 21  # fetch_k (20) + 1


def test_tool_get_recommendations_seed_not_in_catalog(recommender):
    result = recommender.tool_get_recommendations(seed_movie_title="Totally Unknown Film")
    assert result["method"] == "not_found"
    assert result["recommendations"] == []


def test_tool_get_recommendations_popularity_fallback_when_no_input(recommender):
    result = recommender.tool_get_recommendations(top_k=3)
    assert result["method"] == "popularity_fallback"
    # Highest rating counts: movieId 2 (80000), 5 (70000), 6 (65000)
    assert result["recommendations"] == ["Dark Knight, The (2008)", "Toy Story (1995)", "Forrest Gump (1994)"]


def test_tool_get_recommendations_handles_top_k_none_regression(recommender):
    """Regression test: Groq's API sometimes sends top_k=None explicitly
    (rather than omitting it) when the model doesn't want to specify a
    value, which used to crash the slicing logic downstream. It must
    silently fall back to the default of 5 instead of raising."""
    result = recommender.tool_get_recommendations(top_k=None)
    assert result["method"] == "popularity_fallback"
    assert len(result["recommendations"]) == 5  # default, not an error


def test_tool_get_recommendations_unknown_user_falls_back_to_popularity(recommender):
    result = recommender.tool_get_recommendations(user_id=999999, top_k=2)
    assert result["method"] == "popularity_fallback"


# ---------------------------------------------------------------------------
# run_agent — the tool-calling orchestration
# ---------------------------------------------------------------------------

class _FakeToolCall:
    def __init__(self, name, arguments):
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def _fake_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_run_agent_no_tool_call_path(recommender):
    """When the model doesn't call a tool (e.g. an off-topic message), the
    reply should pass through directly and exactly one assistant message
    should be appended — no duplication."""
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(
        _FakeMessage(content="I can only help with movies!")
    )

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What's the weather today?"},
    ]
    reply, updated_history, tool_info = run_agent(client, recommender, history)

    assert reply == "I can only help with movies!"
    assert tool_info == []
    assert updated_history[-1] == {"role": "assistant", "content": "I can only help with movies!"}
    assert len(updated_history) == 3  # system + user + assistant, nothing duplicated


def test_run_agent_tool_call_path_calls_the_right_tool(recommender):
    """When the model calls a tool, run_agent should execute it against the
    real MovieRecommender instance (not fabricate a result) and produce a
    final natural-language reply from a second completion call."""
    client = MagicMock()
    tool_call_response = _fake_response(
        _FakeMessage(tool_calls=[_FakeToolCall("tool_lookup_movie_info", {"title_query": "Toy Story"})])
    )
    final_response = _fake_response(_FakeMessage(content="Toy Story is an animated comedy!"))
    client.chat.completions.create.side_effect = [tool_call_response, final_response]

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What genre is Toy Story?"},
    ]
    reply, updated_history, tool_info = run_agent(client, recommender, history)

    assert reply == "Toy Story is an animated comedy!"
    assert len(tool_info) == 1
    assert tool_info[0]["tool"] == "tool_lookup_movie_info"
    assert tool_info[0]["result"]["method"] == "lookup"  # real tool output, not fabricated
    assert client.chat.completions.create.call_count == 2


def test_run_agent_does_not_duplicate_or_lose_the_user_message_regression(recommender):
    """Regression test for a real bug: run_agent previously referenced an
    undefined `user_message` variable when building the follow-up prompt
    after a tool call (a leftover from an earlier function signature). This
    confirms the follow-up correctly reads the user's message FROM
    conversation_history instead, and that exactly one new message
    (the assistant's reply) is appended — the user's turn is never
    duplicated or dropped.
    """
    client = MagicMock()
    tool_call_response = _fake_response(
        _FakeMessage(tool_calls=[_FakeToolCall("tool_lookup_movie_info", {"title_query": "Toy Story"})])
    )
    final_response = _fake_response(_FakeMessage(content="It's a comedy!"))
    client.chat.completions.create.side_effect = [tool_call_response, final_response]

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What genre is Toy Story?"},
    ]
    original_length = len(history)

    _, updated_history, _ = run_agent(client, recommender, history)

    assert len(updated_history) == original_length + 1
    assert updated_history[-1]["role"] == "assistant"


def test_tool_schemas_allow_null_for_optional_params_regression():
    """Regression test: Groq's API strictly validates JSON schema types and
    rejects a tool call where an unused optional parameter is explicitly
    passed as null, unless the schema allows it. Each optional parameter
    must declare ['type', 'null'] rather than a single bare type.
    """
    props = TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]
    for param in ["user_id", "seed_movie_title", "top_k"]:
        assert "null" in props[param]["type"], (
            f"'{param}' must allow null — Groq rejects explicit nulls "
            "against a single-type schema."
        )