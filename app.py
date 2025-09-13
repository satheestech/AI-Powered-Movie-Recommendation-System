
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 AI-Powered Movie Recommendation System (Level-2)")
st.markdown(
    "Search a movie (TMDB), select it, and get content-based recommendations (posters, title, genres)."
)

# TMDB API key (read from env or secret or user input)
import os
tmdb_api_key = os.environ.get("TMDB_API_KEY") or st.text_input(
    "Enter your TMDB API key (get one at https://www.themoviedb.org/settings/api).", type="password"
)
if not tmdb_api_key:
    st.warning("A TMDB API key is required. Obtain one at https://www.themoviedb.org/.")
    st.stop()

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

def tmdb_get(path: str, params: dict = None) -> dict:
    if params is None:
        params = {}
    params.update({"api_key": tmdb_api_key, "language": "en-US"})
    resp = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(show_spinner=False)
def search_movies(query: str, max_results: int = 10):
    results = []
    page = 1
    while len(results) < max_results:
        data = tmdb_get("/search/movie", {"query": query, "page": page, "include_adult": False})
        results.extend(data.get("results", []))
        if page >= data.get("total_pages", 1):
            break
        page += 1
    return results[:max_results]

@st.cache_data(show_spinner=False)
def get_movie_details(movie_id: int):
    return tmdb_get(f"/movie/{movie_id}", {"append_to_response": "credits"})

@st.cache_data(show_spinner=False)
def fetch_corpus(max_movies: int = 800):
    movies = {}
    endpoints = ["/movie/popular", "/movie/top_rated", "/movie/upcoming", "/movie/now_playing"]
    for endpoint in endpoints:
        page = 1
        while len(movies) < max_movies:
            try:
                data = tmdb_get(endpoint, {"page": page})
            except Exception:
                break
            for m in data.get("results", []):
                mid = m["id"]
                if mid in movies:
                    continue
                movies[mid] = {
                    "movie_id": mid,
                    "title": m.get("title") or m.get("name"),
                    "overview": m.get("overview") or "",
                    "poster_path": m.get("poster_path"),
                    "genre_ids": m.get("genre_ids", []),
                }
            if page >= data.get("total_pages", 1):
                break
            page += 1
            if len(movies) >= max_movies:
                break
        if len(movies) >= max_movies:
            break

    try:
        genre_map_resp = tmdb_get("/genre/movie/list")
        genre_map = {g["id"]: g["name"] for g in genre_map_resp.get("genres", [])}
    except Exception:
        genre_map = {}

    rows = []
    for m in movies.values():
        genre_names = [genre_map.get(gid, "") for gid in m.get("genre_ids", [])]
        rows.append({
            "movie_id": m["movie_id"],
            "title": m["title"],
            "overview": m["overview"],
            "genres": ", ".join([gn for gn in genre_names if gn]),
            "poster_path": m["poster_path"]
        })
    return pd.DataFrame(rows)

@st.cache_resource
def build_vectorizer_and_matrix(df: pd.DataFrame):
    def make_text(row):
        g = row.get("genres") or ""
        o = row.get("overview") or ""
        return f"{g} " + o
    texts = df.apply(make_text, axis=1).fillna("")
    vec = TfidfVectorizer(stop_words="english", max_features=20000)
    mat = vec.fit_transform(texts)
    return vec, mat

def recommend(selected_movie_id: int, corpus_df: pd.DataFrame, vec, mat, top_k: int = 6):
    try:
        idx = corpus_df.index[corpus_df["movie_id"] == selected_movie_id].tolist()[0]
    except IndexError:
        return []
    query_vec = mat[idx]
    cosine_similarities = linear_kernel(query_vec, mat).flatten()
    cosine_similarities[idx] = -1
    top_indices = cosine_similarities.argsort()[::-1][:top_k]
    results = []
    for i in top_indices:
        if cosine_similarities[i] <= 0:
            continue
        results.append({
            "movie_id": int(corpus_df.iloc[i]["movie_id"]),
            "title": corpus_df.iloc[i]["title"],
            "overview": corpus_df.iloc[i]["overview"],
            "genres": corpus_df.iloc[i]["genres"],
            "poster_path": corpus_df.iloc[i]["poster_path"],
            "score": float(cosine_similarities[i])
        })
    return results

st.markdown("### Search for a movie")
query = st.text_input("Type a movie title and press Enter")
if not query:
    st.info("Enter a movie title to begin.")
    st.stop()

with st.spinner("Searching TMDB..."):
    try:
        search_results = search_movies(query, max_results=10)
    except Exception as e:
        st.error(f"TMDB search failed: {e}")
        st.stop()

if not search_results:
    st.warning("No results found.")
    st.stop()

options = []
id_map = {}
for r in search_results:
    title = r.get("title") or r.get("name")
    year = (r.get("release_date") or "")[:4]
    display = f"{title} ({year}) — id:{r['id']}"
    options.append(display)
    id_map[display] = r["id"]

selected_display = st.selectbox("Select the exact movie", options)
selected_id = id_map[selected_display]

with st.spinner("Fetching details..."):
    try:
        details = get_movie_details(selected_id)
    except Exception as e:
        st.error(f"Could not fetch movie details: {e}")
        st.stop()

selected_title = details.get("title") or details.get("name")
selected_genres = ", ".join([g["name"] for g in details.get("genres", [])])
selected_overview = details.get("overview", "")

st.markdown("#### Selected Movie")
col1, col2 = st.columns([1, 2])
with col1:
    poster_path = details.get("poster_path")
    if poster_path:
        st.image(f"{TMDB_IMAGE_BASE}{poster_path}", use_column_width=True)
    else:
        st.write("No poster available")
with col2:
    st.write(f"**{selected_title}**")
    st.write(f"**Genres:** {selected_genres or 'N/A'}")
    st.write(f"**Overview:** {selected_overview or 'N/A'}")

max_corpus_movies = st.sidebar.slider("Corpus size", min_value=100, max_value=1200, value=600, step=100)
with st.spinner("Building corpus..."):
    try:
        corpus_df = fetch_corpus(max_movies=max_corpus_movies)
    except Exception as e:
        st.error(f"Failed to build corpus: {e}")
        st.stop()

if selected_id not in set(corpus_df["movie_id"].astype(int)):
    corpus_df = pd.concat([corpus_df, pd.DataFrame([{
        "movie_id": selected_id,
        "title": selected_title,
        "overview": selected_overview or "",
        "genres": selected_genres or "",
        "poster_path": details.get("poster_path")
    }])], ignore_index=True)

with st.spinner("Vectorizing and computing recommendations..."):
    vec, mat = build_vectorizer_and_matrix(corpus_df)
    recs = recommend(selected_id, corpus_df, vec, mat, top_k=6)

if not recs:
    st.warning("No strong recommendations found.")
else:
    st.markdown("### Recommended Movies")
    cols = st.columns(3)
    for i, rec in enumerate(recs[:6]):
        col = cols[i % 3]
        with col:
            poster = rec.get("poster_path")
            if poster:
                st.image(f"{TMDB_IMAGE_BASE}{poster}", use_column_width=True)
            else:
                st.write("(No poster)")
            st.markdown(f"**{rec['title']}**")
            st.write(f"Genres: {rec['genres'] or 'N/A'}")
            st.caption(f"Similarity score: {rec['score']:.3f}")
