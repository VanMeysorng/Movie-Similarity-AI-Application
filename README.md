# Movie-Similarity-AI-Application

** Overview
>> Movie Similarity Finder is a Python application that computes and displays movie similarities based on user ratings from the MovieLens dataset. It leverages Apache Spark for efficient computation and Streamlit for an interactive user interface.

** Features
>> - Load Movie Names: Load movie names from the dataset.
>> - Make Pairs: Create pairs of movie ratings for similarity calculation.
>> - Filter Duplicates: Filter out duplicate movie pairs.
>> - Compute Cosine Similarity: Compute cosine similarity for pairs of movie ratings.

** Setup Environment

>> - Ensure PySpark is installed (If not, install it using pip install pyspark).
>> - Download and extract the MovieLens 100k dataset into a directory, e.g., SparkCourse/ml-100k/.

** Usage
>> Launch the application using Streamlit: 
>> **streamlit run main.py**
