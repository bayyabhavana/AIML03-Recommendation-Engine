
import pandas as pd
import os

print("STARTING DATA CLEANING")

# Step 1: Create output folder
os.makedirs("data/processed", exist_ok=True)

# Step 2: Load ratings
ratings = pd.read_csv(
    "data/ml-25m/ratings.csv",
    nrows=100000
)

# Step 3: Load movies
movies = pd.read_csv(
    "data/ml-25m/movies.csv"
)

print("\nOriginal ratings:", len(ratings))
print("Original movies:", len(movies))

# Step 4: Check missing values
print("\nMissing values:")
print(ratings.isnull().sum())

# Step 5: Remove duplicate ratings
ratings = ratings.drop_duplicates()

# Step 6: Remove rows with missing values
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating", "timestamp"]
)

# Step 7: Keep valid ratings
ratings = ratings[
    (ratings["rating"] >= 0.5) &
    (ratings["rating"] <= 5.0)
]

# Step 8: Remove duplicate movies
movies = movies.drop_duplicates(
    subset=["movieId"]
)

# Step 9: Save cleaned data
ratings.to_csv(
    "data/processed/clean_ratings.csv",
    index=False
)

movies.to_csv(
    "data/processed/clean_movies.csv",
    index=False
)

print("\nCLEANING COMPLETED")

print("Clean ratings:", len(ratings))
print("Clean movies:", len(movies))

print("\nCleaned files saved successfully!")
