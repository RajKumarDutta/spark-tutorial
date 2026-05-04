import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, expr, rand
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, Normalizer
from pyspark.ml.functions import vector_to_array
import hashlib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Netflix Clone", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
def local_css():
    st.markdown("""
        <style>
        .stApp { background-color: #141414; color: white; }
        h1, h2, h3, p, span, label { color: white !important; }

        .movie-card {
            position: relative;
            transition: transform 0.3s ease;
            border-radius: 4px;
            overflow: hidden;
            background: #2f2f2f;
            aspect-ratio: 2/3;
        }
        .movie-card:hover { transform: scale(1.05); }
        .movie-img { width: 100%; height: 100%; object-fit: cover; }

        .movie-title-overlay {
            position: absolute;
            bottom: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.9));
            width: 100%;
            padding: 10px;
            font-size: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)


# ----------------------------
# Poster Generator
# ----------------------------
def get_poster_url(title):
    seed = int(hashlib.md5(title.encode()).hexdigest(), 16) % 1000
    return f"https://picsum.photos/300/450?random={seed}"


# ----------------------------
# Spark Session (RESOURCE CACHE)
# ----------------------------
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("NetflixRecommender") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()


# ----------------------------
# Load Data (RESOURCE CACHE)
# ----------------------------
@st.cache_resource
def load_and_prep_data():
    spark = get_spark()

    try:
        df = spark.read.csv("netflix_titles.csv", header=True, inferSchema=True)
    except:
        st.error("❌ netflix_titles.csv not found")
        return None

    df = df.fillna("")

    df = df.withColumn(
        "combined",
        concat_ws(" ", col("title"), col("listed_in"), col("description"), col("cast"))
    )

    tokenizer = Tokenizer(inputCol="combined", outputCol="words")
    wordsData = tokenizer.transform(df)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
    final_df = normalizer.transform(rescaledData)

    final_df = final_df.withColumn("features_array", vector_to_array(col("normFeatures")))

    return final_df


# ----------------------------
# Recommendation Logic
# ----------------------------
def get_recommendations(movie_name, df, top_n=6):
    target = df.filter(col("title") == movie_name).select("features_array").first()

    if not target:
        return []

    target_vec = target["features_array"]

    expr_str = "aggregate(zip_with(features_array, array({}), (x, y) -> x * y), 0D, (acc, x) -> acc + x)".format(
        ",".join(map(str, target_vec))
    )

    recs = df.withColumn("similarity", expr(expr_str)) \
        .filter(col("title") != movie_name) \
        .orderBy(col("similarity").desc()) \
        .select("title", "listed_in") \
        .limit(top_n)

    return recs.collect()


# ----------------------------
# UI Components
# ----------------------------
def movie_card(title, category):
    poster = get_poster_url(title)

    st.markdown(f"""
        <div class="movie-card">
            <img src="{poster}" class="movie-img" />
            <div class="movie-title-overlay">
                <div>{title}</div>
                <div style="color:#aaa;">{category}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ----------------------------
# Main App
# ----------------------------
def main():
    local_css()

    data = load_and_prep_data()
    if data is None:
        return

    # Sidebar
    st.sidebar.title("🎬 Netflix Recommender")

    # ⚠️ Limit data fetch (important)
    titles = [row["title"] for row in data.select("title").distinct().limit(1000).collect()]
    query = st.sidebar.selectbox("Search Movie", [""] + sorted(titles))

    # Hero
    st.markdown("""
        <div style="padding:40px;">
            <h1>Netflix Recommender</h1>
            <p>Find movies similar to what you love</p>
        </div>
    """, unsafe_allow_html=True)

    # Recommendations
    if query:
        st.subheader(f"Recommendations for '{query}'")

        recs = get_recommendations(query, data)
        cols = st.columns(6)

        for i, row in enumerate(recs):
            with cols[i]:
                movie_card(row["title"], row["listed_in"])

    # Random Section
    st.markdown("### Popular")
    popular = data.orderBy(rand()).limit(6).collect()

    cols = st.columns(6)
    for i, row in enumerate(popular):
        with cols[i]:
            movie_card(row["title"], row["listed_in"])


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    main()