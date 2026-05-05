import streamlit as st
import pandas as pd
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Netflix Clone", layout="wide")

# ----------------------------
# CSS
# ----------------------------
def local_css():
    st.markdown("""
        <style>
        .stApp { background-color: #141414; color: white; }

        .movie-card {
            position: relative;
            border-radius: 4px;
            overflow: hidden;
            background: #2f2f2f;
        }

        .movie-img { width: 100%; height: 100%; }

        .movie-title-overlay {
            position: absolute;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            width: 100%;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


# ----------------------------
# Poster
# ----------------------------
def get_poster_url(title):
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16) % 1000
    return f"https://picsum.photos/300/450?random={seed}"


# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df = df.fillna("")

    df["combined"] = (
        df["title"] + " " +
        df["listed_in"] + " " +
        df["description"] + " " +
        df["cast"]
    )

    return df


# ----------------------------
# Build Similarity
# ----------------------------
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    similarity = cosine_similarity(matrix)
    return similarity


# ----------------------------
# Recommendation
# ----------------------------
def get_recommendations(title, df, sim_matrix):
    if title not in df["title"].values:
        return []

    idx = df[df["title"] == title].index[0]

    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:7]

    return df.iloc[[i[0] for i in scores]][["title", "listed_in"]]


# ----------------------------
# UI Card
# ----------------------------
def movie_card(title, category):
    poster = get_poster_url(title)

    st.markdown(f"""
        <div class="movie-card">
            <img src="{poster}" class="movie-img"/>
            <div class="movie-title-overlay">
                <div>{title}</div>
                <div style="color:#aaa;">{category}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ----------------------------
# Main
# ----------------------------
def main():
    local_css()

    df = load_data()
    sim_matrix = build_model(df)

    st.sidebar.title("🎬 Netflix Recommender")

    titles = df["title"].drop_duplicates().tolist()
    query = st.sidebar.selectbox("Search Movie", [""] + sorted(titles))

    st.markdown("<h1>Netflix Recommender</h1>", unsafe_allow_html=True)

    if query:
        recs = get_recommendations(query, df, sim_matrix)

        cols = st.columns(6)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i]:
                movie_card(row["title"], row["listed_in"])


if __name__ == "__main__":
    main()