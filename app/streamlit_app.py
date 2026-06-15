"""Streamlit app for exploring movie recommendations."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.baseline import PopularityRecommender
from src.item_cf import ItemCFRecommender
from src.svd_model import SVDRecommender

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:wght@600&display=swap');

    :root {
        --bg-1: #081320;
        --bg-2: #0e1d30;
        --bg-3: #132741;
        --accent: #d4a447;
        --card: rgba(20, 35, 55, 0.78);
        --card-strong: rgba(25, 42, 66, 0.92);
        --text: #edf2fb;
        --muted: #b9c5d8;
        --border: rgba(255, 255, 255, 0.11);
        --accent-border: rgba(212, 164, 71, 0.34);
        --shadow: 0 14px 36px rgba(2, 8, 18, 0.42);
    }

    .stApp {
        background:
            radial-gradient(1200px 620px at -15% -10%, #1a3357 0%, rgba(8, 19, 32, 0) 60%),
            radial-gradient(1000px 560px at 110% 0%, #1a2f4f 0%, rgba(14, 29, 48, 0) 58%),
            linear-gradient(165deg, var(--bg-1) 0%, var(--bg-2) 55%, var(--bg-3) 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: var(--text);
        font-family: 'Fraunces', serif;
        letter-spacing: 0.2px;
    }

    p, label, .stCaption {
        color: var(--muted);
    }

    .subtle {
        color: var(--muted);
    }

    .accent {
        color: var(--accent);
        font-weight: 700;
    }

    .hero-card {
        background: linear-gradient(155deg, rgba(27, 45, 69, 0.92), rgba(14, 24, 40, 0.92));
        border-radius: 18px;
        padding: 1.05rem 1.2rem;
        border: 1px solid var(--accent-border);
        box-shadow: var(--shadow);
        margin-bottom: 0.9rem;
        position: relative;
        overflow: hidden;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        right: -100px;
        top: -110px;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(212, 164, 71, 0.18), rgba(212, 164, 71, 0));
        pointer-events: none;
    }

    .hero-kicker {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 1.1px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.28rem;
    }

    .hero-title {
        color: var(--text);
        font-family: 'Fraunces', serif;
        font-size: clamp(1.45rem, 2.1vw, 2.2rem);
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: var(--muted);
        margin: 0;
        max-width: 72ch;
    }

    .card {
        background: var(--card);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border: 1px solid var(--border);
        box-shadow: 0 10px 28px rgba(3, 10, 20, 0.3);
        margin-bottom: 0.75rem;
    }

    .section-card {
        background: var(--card-strong);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.7rem 0.95rem;
        margin: 0.15rem 0 0.7rem;
        box-shadow: 0 8px 20px rgba(2, 7, 15, 0.3);
    }

    .section-title {
        color: var(--text);
        font-weight: 650;
        font-size: 1.02rem;
        margin: 0;
    }

    .section-note {
        color: var(--muted);
        margin: 0.2rem 0 0;
        font-size: 0.87rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(17, 30, 47, 0.98), rgba(11, 21, 34, 0.98));
        border-right: 1px solid rgba(212, 164, 71, 0.18);
    }

    .sidebar-panel {
        background: rgba(212, 164, 71, 0.08);
        border: 1px solid rgba(212, 164, 71, 0.28);
        border-radius: 12px;
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.7rem;
    }

    .sidebar-title {
        color: #f6dfad;
        font-size: 0.94rem;
        font-weight: 650;
        margin: 0;
    }

    .sidebar-subtitle {
        color: var(--muted);
        margin: 0.24rem 0 0;
        font-size: 0.82rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.09);
        padding-bottom: 0.35rem;
        margin-bottom: 0.7rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 999px;
        color: #c8d3e6;
        font-weight: 600;
        padding: 0.34rem 0.9rem;
        transition: all 0.18s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(212, 164, 71, 0.15) !important;
        border-color: rgba(212, 164, 71, 0.54) !important;
        color: #f7e4ba !important;
    }

    .stTextInput > div > div,
    .stSelectbox > div > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(13, 24, 39, 0.93) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }

    .stTextInput > div > div:focus-within,
    .stSelectbox > div > div:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: rgba(212, 164, 71, 0.65) !important;
        box-shadow: 0 0 0 1px rgba(212, 164, 71, 0.28);
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        border: 2px solid rgba(212, 164, 71, 0.92) !important;
        background: #0d1828 !important;
        box-shadow: 0 0 0 4px rgba(212, 164, 71, 0.12);
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background: rgba(212, 164, 71, 0.3);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(212, 164, 71, 0.48);
        background: linear-gradient(180deg, rgba(212, 164, 71, 0.2), rgba(212, 164, 71, 0.14));
        color: #f7e4bd;
        font-weight: 620;
    }

    [data-testid="stDataFrame"],
    .stTable {
        background: var(--card-strong);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.22rem;
        box-shadow: 0 8px 22px rgba(3, 9, 18, 0.36);
    }

    .stAlert {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.14);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_processed_tables(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(data_dir) / "processed"
    ratings_path = processed_dir / "ratings_clean.csv"
    movies_path = processed_dir / "movies_clean.csv"
    if not ratings_path.exists() or not movies_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies


@st.cache_resource
def load_item_model(ratings: pd.DataFrame) -> ItemCFRecommender:
    return ItemCFRecommender(min_ratings=20).fit(ratings)


@st.cache_resource
def load_svd_model(ratings: pd.DataFrame) -> SVDRecommender:
    model_path = Path("outputs") / "models" / "svd_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return SVDRecommender(n_factors=100, random_state=42).fit(ratings)


def movie_label(row: pd.Series) -> str:
    year = f" ({int(row['year'])})" if "year" in row and pd.notna(row["year"]) else ""
    return f"{row['title']}{year}"


st.markdown(
    """
    <div class='hero-card'>
        <div class='hero-kicker'>Movie Analytics Dashboard</div>
        <div class='hero-title'>Movie Recommender System</div>
        <p class='hero-subtitle'>Explore popular titles, item-to-item similarity, and personalized rating predictions with clear, component-based layout.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='card'>
        <h3>Explore three views</h3>
        <p class='subtle'>Popular titles, item-to-item similarity, and personalized rating predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-panel'>
            <p class='sidebar-title'>Setup</p>
            <p class='sidebar-subtitle'>Configure data path and recommendation size</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Setup")
    data_dir = st.text_input("Data directory", value="data")
    top_n = st.slider("Top-N", min_value=5, max_value=20, value=10, step=1)

ratings, movies = load_processed_tables(data_dir)
if ratings.empty or movies.empty:
    st.error("Processed files not found. Run Phase 2 to generate data/processed files.")
    st.stop()

tabs = st.tabs(["Popular", "Similar", "Predict"])

with tabs[0]:
    st.markdown(
        """
        <div class='section-card'>
            <p class='section-title'>Popular picks</p>
            <p class='section-note'>Top-ranked movies after applying a minimum rating-count threshold.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Popular picks")
    min_ratings = st.slider(
        "Minimum ratings", min_value=10, max_value=200, value=50, step=10
    )
    model = PopularityRecommender(min_ratings=min_ratings).fit(ratings, movies)
    top_movies = model.recommend(top_n=top_n)
    st.dataframe(top_movies, use_container_width=True)

with tabs[1]:
    st.markdown(
        """
        <div class='section-card'>
            <p class='section-title'>Movies like this</p>
            <p class='section-note'>Nearest neighbors from item-item collaborative filtering.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Movies like this")
    movies = movies.copy()
    item_model = load_item_model(ratings)

    eligible_movie_ids = set(item_model.similarity_matrix.index.tolist())
    eligible_movies = movies[movies["movieId"].isin(eligible_movie_ids)].copy()
    eligible_movies["label"] = eligible_movies.apply(movie_label, axis=1)

    movie_choice = st.selectbox(
        "Pick a movie", options=eligible_movies["label"].tolist()
    )
    selected_id = int(
        eligible_movies.loc[eligible_movies["label"] == movie_choice, "movieId"].iloc[0]
    )

    recs = item_model.recommend(selected_id, top_n=top_n)
    recs = recs.merge(movies, on="movieId", how="left")
    st.dataframe(recs, use_container_width=True)

with tabs[2]:
    st.markdown(
        """
        <div class='section-card'>
            <p class='section-title'>Predict a rating</p>
            <p class='section-note'>Estimated score from matrix factorization (SVD).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Predict a rating")
    user_ids = sorted(ratings["userId"].unique().tolist())
    movie_ids = movies["movieId"].tolist()
    user_id = st.selectbox("User", options=user_ids)
    movie_id = st.selectbox(
        "Movie",
        options=movie_ids,
        format_func=lambda mid: movie_label(movies[movies["movieId"] == mid].iloc[0]),
    )
    svd_model = load_svd_model(ratings)
    pred = svd_model.predict(int(user_id), int(movie_id))
    st.markdown(
        f"**Predicted rating:** <span class='accent'>{pred:.2f}</span>",
        unsafe_allow_html=True,
    )
    known = ratings[(ratings["userId"] == user_id) & (ratings["movieId"] == movie_id)]
    if not known.empty:
        actual = float(known.iloc[0]["rating"])
        st.markdown(
            f"<span class='subtle'>Actual rating in data:</span> {actual:.1f}",
            unsafe_allow_html=True,
        )
