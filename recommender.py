"""
recommender.py

Standalone version of the recommendation + agent-tool logic originally built
in the Kaggle notebook (Phases 2 and 3). This file has no Streamlit-specific
code in it — it can be imported by app.py or used/tested on its own.

Requires two files, produced by the notebook:
  - movies_clean.csv   (from Phase 1)
  - als_model.pkl      (from Phase 2 — contains model, user_to_idx,
                         movie_to_idx, idx_to_movie, user_item_matrix)
"""

import re
import json
import pickle
import difflib

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


class MovieRecommender:
    """Loads the trained ALS model + catalog once, and exposes the same
    tool functions used by the LLM agent in the notebook.

    Model/catalog files are large (the ALS model's pickle can be 300MB+,
    well over GitHub's 100MB file limit), so they're hosted on Hugging Face
    Hub instead of committed to the repo. hf_hub_download fetches and caches
    them locally on first use — subsequent calls reuse the cached copy."""

    def __init__(self, hf_repo_id: str, movies_csv_filename: str = "movies_clean.csv",
                 model_pkl_filename: str = "als_model.pkl"):
        movies_csv_path = hf_hub_download(repo_id=hf_repo_id, filename=movies_csv_filename)
        model_pkl_path = hf_hub_download(repo_id=hf_repo_id, filename=model_pkl_filename)

        self.movies = pd.read_csv(movies_csv_path)

        with open(model_pkl_path, "rb") as f:
            artifacts = pickle.load(f)

        self.model = artifacts["model"]
        self.user_to_idx = artifacts["user_to_idx"]
        self.movie_to_idx = artifacts["movie_to_idx"]
        self.idx_to_movie = artifacts["idx_to_movie"]
        self.user_item_matrix = artifacts["user_item_matrix"]

        # Reconstruct per-movie rating counts directly from the matrix —
        # number of non-zero entries in each column = number of ratings
        # that movie received. This avoids needing to re-save `train`
        # from the notebook; the pickle already has everything required.
        counts_by_idx = np.asarray(self.user_item_matrix.getnnz(axis=0)).flatten()
        movie_ids_in_order = [self.idx_to_movie[i] for i in range(len(counts_by_idx))]
        self.movie_rating_counts = pd.Series(counts_by_idx, index=movie_ids_in_order)

        self.popularity_ranking = self.movie_rating_counts.sort_values(ascending=False).index.tolist()

        # Precompute cleaned titles once for fast fuzzy matching later
        self.movie_titles = self.movies["title"].tolist()
        self.movie_titles_clean = [
            self._normalize_article(self._strip_year(t)) for t in self.movie_titles
        ]

    # ---------- hybrid recommendation: blend ALS similarity with genre overlap ----------
    #
    # Pure ALS similarity is behavior-based only ("people who liked A also
    # liked B"), which sometimes surfaces thematically unrelated movies that
    # happen to share an audience (e.g. a prestige drama next to an action
    # movie). Blending in genre overlap doesn't replace ALS — it re-ranks a
    # larger pool of ALS candidates so thematically consistent picks are
    # preferred when the raw ALS scores are close.

    HYBRID_ALS_WEIGHT = 0.5  # 0.5 = equal weight to behavior (ALS) and content (genre)

    @staticmethod
    def _genre_set(genres_str) -> set:
        return set(genres_str.split("|")) if genres_str else set()

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        union = len(set_a | set_b)
        return len(set_a & set_b) / union if union else 0.0

    def _genres_for(self, movie_id) -> set:
        # Lazily build and cache a movieId -> genre-set lookup on first use,
        # so repeated calls (one per candidate being re-ranked) are O(1)
        # instead of scanning the movies dataframe each time.
        if not hasattr(self, "_genre_lookup_cache"):
            self._genre_lookup_cache = dict(zip(self.movies["movieId"], self.movies["genres"]))
        return self._genre_set(self._genre_lookup_cache.get(movie_id, ""))

    # ---------- genre/year filtering ----------
    #
    # This is a HARD constraint ("must be this genre / from this era"),
    # separate from the soft genre-overlap SCORE used for hybrid re-ranking
    # above. A request like "suggest some thriller movies" needs a real
    # filter — without one, the agent could only fall back to generic
    # popularity and either present irrelevant results or awkwardly admit
    # it has nothing to offer, both of which were observed in real testing.

    @staticmethod
    def _normalize_genre_token(g: str) -> str:
        return re.sub(r"[^a-z0-9]", "", g.lower())

    def _genre_matches(self, movie_id, genre_query: str) -> bool:
        query_norm = self._normalize_genre_token(genre_query)
        return any(
            query_norm in self._normalize_genre_token(g) or self._normalize_genre_token(g) in query_norm
            for g in self._genres_for(movie_id)
        )

    def _year_for(self, movie_id):
        if not hasattr(self, "_year_lookup_cache"):
            def extract_year(title):
                m = re.search(r"\((\d{4})\)\s*$", title)
                return int(m.group(1)) if m else None
            self._year_lookup_cache = {
                mid: extract_year(title) for mid, title in zip(self.movies["movieId"], self.movies["title"])
            }
        return self._year_lookup_cache.get(movie_id)

    def _passes_filters(self, movie_id, genre=None, min_year=None, max_year=None) -> bool:
        if genre and not self._genre_matches(movie_id, genre):
            return False
        if min_year is not None or max_year is not None:
            year = self._year_for(movie_id)
            if year is None:
                return False  # can't confirm it meets a year constraint, so exclude it
            if min_year is not None and year < min_year:
                return False
            if max_year is not None and year > max_year:
                return False
        return True

    # ---------- title matching helpers ----------

    @staticmethod
    def _strip_year(title: str) -> str:
        return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip().lower()

    @staticmethod
    def _normalize_article(title: str) -> str:
        """Move/strip leading or trailing 'the'/'a'/'an' so 'The Dark Knight'
        matches 'Dark Knight, The'."""
        title = re.sub(r",\s*(the|a|an)$", "", title)
        title = re.sub(r"^(the|a|an)\s+", "", title)
        return title.strip()

    def find_movie_id(self, title_query: str, cutoff: float = 0.6):
        """Match a user-typed title to the closest real movie in the catalog,
        preferring the most-rated match when multiple titles fit."""
        query_clean = self._normalize_article(self._strip_year(title_query))
        query_clean = re.sub(r"[!?.]+", "", query_clean).strip()

        def pick_most_popular(indices):
            candidates = self.movies.iloc[indices][["movieId"]].copy()
            candidates["rating_count"] = candidates["movieId"].map(self.movie_rating_counts).fillna(0)
            best_pos = candidates["rating_count"].values.argmax()
            best_idx = indices[best_pos]
            return self.movies.iloc[best_idx]["movieId"], self.movie_titles[best_idx]

        exact_hits = [i for i, t in enumerate(self.movie_titles_clean) if query_clean == t]
        if exact_hits:
            return pick_most_popular(exact_hits)

        substring_hits = [i for i, t in enumerate(self.movie_titles_clean) if query_clean in t]
        if substring_hits:
            return pick_most_popular(substring_hits)

        matches = difflib.get_close_matches(query_clean, self.movie_titles_clean, n=1, cutoff=cutoff)
        if not matches:
            return None, None

        idx = self.movie_titles_clean.index(matches[0])
        return self.movies.iloc[idx]["movieId"], self.movie_titles[idx]

    # ---------- recommendation logic ----------

    def als_recommend(self, user_id: int, k: int = 10):
        if user_id not in self.user_to_idx:
            return self.popularity_recommend(top_k=k)

        u_idx = self.user_to_idx[user_id]
        ids, scores = self.model.recommend(
            u_idx, self.user_item_matrix[u_idx], N=k, filter_already_liked_items=True
        )
        return [self.idx_to_movie[i] for i in ids]

    def popularity_recommend(self, top_k: int = 10):
        return self.popularity_ranking[:top_k]

    # ---------- tool functions (same interface the LLM agent calls) ----------

    def tool_get_recommendations(
        self,
        user_id: int = None,
        seed_movie_title: str = None,
        top_k: int = 5,
        genre: str = None,
        min_year: int = None,
        max_year: int = None,
    ):
        """
        Get movie recommendations, optionally filtered by genre and/or release year range.
        - If user_id is known, use personalized ALS recommendations.
        - Else if a seed_movie_title is given, recommend similar movies (item-based via ALS item factors).
        - Else, fall back to popularity.

        Filtering uses a retrieve-then-filter approach: fetch more raw
        candidates than needed, then narrow by genre/year, since neither the
        ALS model nor plain popularity ranking has any concept of genre or
        year by itself.
        """
        if top_k is None:
            top_k = 5

        has_filters = bool(genre or min_year is not None or max_year is not None)
        id_to_title = dict(zip(self.movies["movieId"], self.movies["title"]))

        def build_result(method, movie_ids, **extra):
            result = {"method": method, "recommendations": [id_to_title[mid] for mid in movie_ids], **extra}
            if has_filters:
                result["filters_applied"] = {"genre": genre, "min_year": min_year, "max_year": max_year}
                if not movie_ids:
                    result["note"] = (result.get("note", "") + " No results matched those filters; try loosening genre/year.").strip()
            return result

        if user_id is not None and user_id in self.user_to_idx:
            fetch_k = max(top_k * 6, 30) if has_filters else top_k
            recs = self.als_recommend(user_id, k=fetch_k)
            filtered = [mid for mid in recs if self._passes_filters(mid, genre, min_year, max_year)][:top_k]
            return build_result("personalized_als", filtered)

        if seed_movie_title:
            movie_id, matched_title = self.find_movie_id(seed_movie_title)
            if movie_id is not None and movie_id in self.movie_to_idx:
                m_idx = self.movie_to_idx[movie_id]

                # Over-fetch ALS candidates (more than top_k) so there's a
                # real pool to re-rank by genre overlap and/or hard-filter,
                # not just whatever ALS alone would have returned. Fetch
                # even more when a hard filter is active, since some
                # candidates will get excluded entirely rather than just
                # re-ranked.
                fetch_k = max(top_k * 8, 40) if has_filters else max(top_k * 4, 20)
                similar_ids, als_scores = self.model.similar_items(m_idx, N=fetch_k + 1)
                als_scores = np.asarray(als_scores, dtype=float)

                seed_genres = self._genres_for(movie_id)

                # Normalize ALS scores to 0-1 within this candidate set so
                # they're on a comparable scale to the Jaccard genre score
                # before blending (ALS raw scores aren't bounded to 0-1).
                score_range = als_scores.max() - als_scores.min()
                if score_range > 0:
                    als_scores_norm = (als_scores - als_scores.min()) / score_range
                else:
                    als_scores_norm = np.ones_like(als_scores)

                candidates = []
                for raw_idx, als_score_norm in zip(similar_ids, als_scores_norm):
                    cand_movie_id = self.idx_to_movie[raw_idx]
                    if cand_movie_id == movie_id:
                        continue  # a movie isn't "similar to itself"
                    if not self._passes_filters(cand_movie_id, genre, min_year, max_year):
                        continue
                    genre_score = self._jaccard_similarity(seed_genres, self._genres_for(cand_movie_id))
                    blended = (
                        self.HYBRID_ALS_WEIGHT * als_score_norm
                        + (1 - self.HYBRID_ALS_WEIGHT) * genre_score
                    )
                    candidates.append((cand_movie_id, blended))

                candidates.sort(key=lambda pair: pair[1], reverse=True)
                similar_movie_ids = [mid for mid, _ in candidates[:top_k]]
                return build_result("item_similarity", similar_movie_ids, seed_matched_to=matched_title)
            else:
                return {
                    "method": "not_found",
                    "recommendations": [],
                    "note": f"Couldn't match '{seed_movie_title}' to a movie in the catalog.",
                }

        if has_filters:
            # Scan the FULL popularity ranking (already sorted, already in
            # memory — free to iterate) with early exit, rather than only
            # checking a small top slice. A niche genre (Documentary,
            # Musical, Film-Noir) may have zero matches among the overall
            # most-popular movies but plenty further down the list — an
            # early top-slice-then-filter approach would incorrectly report
            # "no results" for a genre the catalog actually has.
            filtered = []
            for mid in self.popularity_ranking:
                if self._passes_filters(mid, genre, min_year, max_year):
                    filtered.append(mid)
                    if len(filtered) >= top_k:
                        break
        else:
            filtered = self.popularity_recommend(top_k=top_k)
        return build_result("popularity_fallback", filtered)

    def tool_lookup_movie_info(self, title_query: str):
        """Look up factual info (title, genres) for a movie by fuzzy title match."""
        movie_id, matched_title = self.find_movie_id(title_query)
        if movie_id is None:
            return {"method": "lookup_not_found", "found": False, "note": f"No close match found for '{title_query}'."}

        row = self.movies.loc[self.movies["movieId"] == movie_id].iloc[0]
        return {
            "method": "lookup",
            "found": True,
            "matched_title": row["title"],
            "genres": row["genres"],
        }


# ---------- tool schema + system prompt (same as the notebook) ----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "tool_get_recommendations",
            "description": "Get movie recommendations, either personalized for a known user_id, similar to a seed movie the user mentions, or popular movies as a fallback. Supports optional filtering by genre and/or release year range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": ["integer", "null"], "description": "Numeric user ID, only if the user explicitly gave one."},
                    "seed_movie_title": {"type": ["string", "null"], "description": "A movie title the user wants similar recommendations to."},
                    "top_k": {"type": ["integer", "null"], "description": "How many recommendations to return. Defaults to 5 if not specified.", "default": 5},
                    "genre": {"type": ["string", "null"], "description": "Filter results to this genre (e.g. 'Comedy', 'Thriller', 'Sci-Fi'), only if the user specifically asked for a genre."},
                    "min_year": {"type": ["integer", "null"], "description": "Only include movies released in or after this year, if the user asked for something recent/newer/from a certain era."},
                    "max_year": {"type": ["integer", "null"], "description": "Only include movies released in or before this year, if the user asked for something older/classic."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_lookup_movie_info",
            "description": "Look up factual information (genres, matched title) about a specific movie by title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_query": {"type": "string", "description": "The movie title to look up, as typed by the user."},
                },
                "required": ["title_query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a friendly movie recommendation assistant. "
    "You MUST call one of your available tools on every single user turn that asks for a recommendation "
    "or asks a factual question about a movie — never answer from your own knowledge, even for vague "
    "requests like 'I'm bored' or 'surprise me' (call tool_get_recommendations with no arguments in that case) "
    "or factual questions like 'what genre is X' or 'tell me about X' (call tool_lookup_movie_info). "
    "If the user asks for a specific genre (e.g. 'a thriller', 'something funny', 'sci-fi movies') or a time "
    "period (e.g. 'something recent', 'an old classic', 'from the 90s'), pass the genre/min_year/max_year "
    "parameters to tool_get_recommendations rather than guessing from an unfiltered list. "
    "Only skip calling a tool if the user's message is clearly unrelated to movies. "
    "Only state facts (like runtime, year, genre, plot details) that were explicitly returned by a tool call. "
    "If the user asks for something a tool doesn't provide, say you don't have that information rather than guessing. "
    "If a tool result includes a 'note' saying no results matched the filters, tell the user honestly and "
    "suggest loosening their request — never present unrelated results as if they matched. "
    "NEVER recommend a movie title that did not come from a tool call in this conversation. "
    "Keep responses concise and conversational."
)


def run_agent(client, recommender: MovieRecommender, conversation_history,
              model_name: str = "openai/gpt-oss-20b"):
    """Same two-call tool-calling flow used in the notebook, but takes the
    OpenAI/Groq client and a MovieRecommender instance as arguments instead
    of relying on notebook globals.

    IMPORTANT: conversation_history must already end with the user's latest
    turn appended (role='user'). This function does NOT append it itself —
    the caller (app.py) does that immediately, before any network call, so
    that if this call gets interrupted (e.g. Streamlit cancels the run
    because the user clicked again), the pending user turn is never lost and
    the next run can simply retry this same call with the same history,
    instead of duplicating the message or getting stuck.
    """

    available_functions = {
        "tool_get_recommendations": recommender.tool_get_recommendations,
        "tool_lookup_movie_info": recommender.tool_lookup_movie_info,
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=conversation_history,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        tool_results = []
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func_result = available_functions[func_name](**func_args)
            tool_results.append((func_name, func_result))

        results_summary = "\n".join(
            f"Result from {name}: {json.dumps(result)}" for name, result in tool_results
        )

        original_user_message = conversation_history[-1]["content"]
        followup_messages = [
            {"role": "system", "content": SYSTEM_PROMPT + " You already have the tool results below — do not attempt to call any tools. Just respond to the user in plain, friendly text."},
            {"role": "user", "content": original_user_message},
            {"role": "user", "content": f"[System note — tool results, not from the user]:\n{results_summary}\n\nNow reply to the user's original message using this information."},
        ]

        second_response = client.chat.completions.create(
            model=model_name,
            messages=followup_messages,
        )
        final_reply = second_response.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": final_reply})
        tool_info = [{"tool": name, "result": result} for name, result in tool_results]
        return final_reply, conversation_history, tool_info

    else:
        conversation_history.append({"role": "assistant", "content": response_message.content})
        return response_message.content, conversation_history, []