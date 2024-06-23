import math
import streamlit as st
import pandas as pd
from pyspark import SparkConf, SparkContext
from streamlit_option_menu import option_menu

# Initialize Spark Context
def init_spark():
    conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
    sc = SparkContext.getOrCreate(conf=conf)
    return sc

# Load Movie Names
def load_movie_names():
    movie_names = {}
    with open("ml-100k/u.item", encoding='ISO-8859-1') as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]
    return movie_names

# Load Ratings Data
def load_ratings_data(sc):
    lines = sc.textFile("ml-100k/u.data")
    ratings = lines.map(lambda x: x.split()).map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))))
    return ratings

# Make Pairs
def make_pairs(user_ratings):
    (user, ratings) = user_ratings
    ratings = list(ratings)
    pairs = []
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            pairs.append(((ratings[i][0], ratings[j][0]), (ratings[i][1], ratings[j][1])))
    return pairs

# Filter Duplicates
def filter_duplicates(movie_pair):
    movie1, movie2 = movie_pair[0]
    return movie1 < movie2

# Compute Cosine Similarity
def compute_cosine_similarity(rating_pairs):
    num_pairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in rating_pairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        num_pairs += 1

    denominator = math.sqrt(sum_xx) * math.sqrt(sum_yy)
    if denominator == 0:
        score = 0
    else:
        score = sum_xy / float(denominator)
    
    return (score, num_pairs)

# Find Similar Movies
def find_similar_movies(sc, score_threshold=0.97, co_occurrence_threshold=50):
    ratings = load_ratings_data(sc)
    ratings_by_user = ratings.groupByKey()
    movie_pairs = ratings_by_user.flatMap(make_pairs)
    filtered_movie_pairs = movie_pairs.filter(filter_duplicates)
    movie_pair_ratings = filtered_movie_pairs.groupByKey()
    movie_pair_similarities = movie_pair_ratings.mapValues(compute_cosine_similarity).cache()
    
    filtered_results = movie_pair_similarities.filter(
        lambda pairSim: pairSim[1][0] > score_threshold and pairSim[1][1] > co_occurrence_threshold
    )
    
    results = filtered_results.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(False)
    return results

# Get Similar Movies
def get_similar_movies(movie_id, results, movie_names, top_n=10):
    top_similar_movies = []
    for result in results.take(top_n):
        (sim, pair) = result
        similar_movie_id = pair[1] if pair[0] == movie_id else pair[0]
        top_similar_movies.append({
            'Movie ID': similar_movie_id,
            'Movie Name': movie_names[similar_movie_id],
            'Similarity Score': sim[0],
            'Co-occurrence': sim[1]
        })
    return top_similar_movies

# Streamlit UI
st.set_page_config(page_title="Movie Similarity Finder", layout="wide")

# Navigation Bar
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Find Similar Movies", "About"],
        icons=["house", "search", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.title("🎬 Movie Similarity Finder")

    # Displaying an image alongside the title
    image = 'Movie.png'  # Replace with your image file path
    st.image(image, caption='Movie Similarity Finder', use_column_width=True)

    st.header("Welcome to the Ms. Movie Similarity Finder!")
    st.markdown("""
        Discover your next favorite movie effortlessly with our Movie Similarity Finder. This application utilizes user ratings from the MovieLens dataset to recommend movies similar to your favorite picks.
    """)

    st.header("Start Exploring")
    st.markdown("""
        Explore a diverse collection of movies spanning various genres and eras. Whether you're a cinephile or a casual viewer, our goal is to enhance your movie-watching experience with personalized recommendations that match your unique taste.
    """)


# Find Similar Movies Page
if selected == "Find Similar Movies":
    st.title("🔍 Find Similar Movies")
    st.markdown("### Select a movie and find its most similar movies based on user ratings.")

    sc = init_spark()
    if sc is not None:
        movie_names = load_movie_names()

        selected_movie_name = st.selectbox(
            "Select a Movie", options=list(movie_names.values())
        )
        movie_id = next(key for key, value in movie_names.items() if value == selected_movie_name)
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.97)
        co_occurrence_threshold = st.slider("Co-occurrence Threshold", 1, 100, 50)

        if st.button("Find Similar Movies"):
            with st.spinner('Finding similar movies...'):
                results = find_similar_movies(sc, score_threshold, co_occurrence_threshold)
                similar_movies = get_similar_movies(movie_id, results, movie_names)
            
            st.success(f"Top 10 similar movies for {movie_names[movie_id]}:")
            st.write("Here are the top similar movies based on the user ratings:")

            # Display results in a table format
            if similar_movies:
                df = pd.DataFrame(similar_movies)
                st.dataframe(df)
            else:
                st.write("No similar movies found. Try adjusting the thresholds.")
    else:
        st.error("Failed to initialize SparkContext. Please check your Spark installation.")

# About Page
if selected == "About":
    st.title("ℹ️ About")
    st.header("Welcome to Ms. Movie Similarity Finder Webapp")
    st.markdown("""
        Discovering your next favorite movie is made effortless with our Movie Similarity Finder. Our cutting-edge algorithms recommend films based on your preferences, ensuring every viewing experience is tailored just for you.
    """)

    st.header("How it Works")
    st.markdown("""
        Whether you're looking for a film similar in genre, plot, or style, our intuitive interface makes it simple. Just enter the title of a movie you love, and instantly uncover a curated selection of similar movies that match your taste.
    """)

    st.header("Our Mission")
    st.markdown("""
        Explore a vast database of cinematic gems, from timeless classics to the latest blockbusters. Our goal is to enhance your movie-watching journey by providing personalized recommendations that resonate with your unique interests.
    """)

    st.header("Get Started")
    st.markdown("""
        Discover, explore, and enjoy the world of cinema like never before with us. Start exploring today and let us guide you to your next movie night masterpiece.
    """)


# Ensure the SparkContext is stopped when Streamlit exits
def cleanup():
    try:
        sc = SparkContext.getOrCreate()
        if sc is not None:
            sc.stop()
    except Exception as e:
        pass

import atexit
atexit.register(cleanup)
