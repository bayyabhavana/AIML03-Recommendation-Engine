
import pandas as pd
import os

print("STARTING FEATURE ENGINEERING")

# Step 1: Load training data
ratings = pd.read_csv("data/processed/train.csv")

# Step 2: Load movie information
movies = pd.read_csv("data/ml-25m/movies.csv")

# Step 3: Create user features
user_features = ratings.groupby("userId").agg(
    total_ratings=("rating", "count"),
    average_rating=("rating", "mean")
).reset_index()

# Step 4: Create movie features
movie_features = ratings.groupby("movieId").agg(
    total_ratings=("rating", "count"),
    average_rating=("rating", "mean")
).reset_index()

# Step 5: Add movie names and genres
movie_features = movie_features.merge(
    movies[["movieId", "title", "genres"]],
    on="movieId",
    how="left"
)

# Step 6: Sort popular movies
movie_features = movie_features.sort_values(
    by="total_ratings",
    ascending=False
)

# Step 7: Save features
os.makedirs("data/processed", exist_ok=True)

user_features.to_csv(
    "data/processed/user_features.csv",
    index=False
)

movie_features.to_csv(
    "data/processed/movie_features.csv",
    index=False
)

# Step 8: Display results
print("\nUSER FEATURES")
print(user_features.head())

print("\nMOVIE FEATURES")
print(movie_features.head())

print("\nTOTAL USERS:", len(user_features))
print("TOTAL MOVIES:", len(movie_features))

print("\nFEATURE ENGINEERING COMPLETED!")
