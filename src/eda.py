
import pandas as pd
import matplotlib.pyplot as plt
import os

print("================================")
print("MOVIE RECOMMENDATION ENGINE")
print("EXPLORATORY DATA ANALYSIS")
print("================================")

# Step 1: Create output folder
os.makedirs("data/processed", exist_ok=True)

# Step 2: Load cleaned datasets
ratings = pd.read_csv(
    "data/processed/clean_ratings.csv"
)

movies = pd.read_csv(
    "data/processed/clean_movies.csv"
)

# Step 3: Basic dataset information
print("\n1. DATASET INFORMATION")

print("Total ratings:", len(ratings))
print("Total users:", ratings["userId"].nunique())
print("Total movies rated:", ratings["movieId"].nunique())
print("Total movies available:", len(movies))

# Step 4: Calculate average rating
average_rating = ratings["rating"].mean()

print("\n2. AVERAGE MOVIE RATING")
print(round(average_rating, 2))

# Step 5: Rating distribution
print("\n3. RATING DISTRIBUTION")

rating_counts = ratings["rating"].value_counts().sort_index()

print(rating_counts)

plt.figure(figsize=(8, 5))

rating_counts.plot(kind="bar")

plt.title("Movie Rating Distribution")
plt.xlabel("Movie Rating")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "data/processed/rating_distribution.png"
)

plt.show()

# Step 6: Find top 10 popular movies
print("\n4. TOP 10 POPULAR MOVIES")

movie_counts = ratings.groupby("movieId").size()

movie_counts = movie_counts.reset_index(
    name="total_ratings"
)

popular_movies = movie_counts.merge(
    movies,
    on="movieId"
)

popular_movies = popular_movies.sort_values(
    by="total_ratings",
    ascending=False
).head(10)

print(
    popular_movies[["title", "total_ratings"]]
)

# Create popular movies chart
plt.figure(figsize=(10, 6))

plt.barh(
    popular_movies["title"].iloc[::-1],
    popular_movies["total_ratings"].iloc[::-1]
)

plt.title("Top 10 Popular Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie Name")

plt.tight_layout()

plt.savefig(
    "data/processed/top_movies.png"
)

plt.show()

# Step 7: Genre analysis
print("\n5. MOVIE GENRE ANALYSIS")

genre_data = movies["genres"].str.split("|").explode()

genre_counts = genre_data.value_counts()

print(genre_counts.head(10))

plt.figure(figsize=(10, 6))

genre_counts.head(10).plot(kind="bar")

plt.title("Top 10 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "data/processed/genre_distribution.png"
)

plt.show()

# Step 8: Save popular movie information
popular_movies.to_csv(
    "data/processed/popular_movies.csv",
    index=False
)

print("\n================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("Charts and reports saved.")
print("================================")
