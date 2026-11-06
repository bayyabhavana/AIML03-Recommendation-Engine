
import pandas as pd
import os

print("STARTING COLD-START RECOMMENDATION SYSTEM")

# 1. Load datasets
ratings = pd.read_csv("data/processed/train.csv")
movies = pd.read_csv("data/processed/clean_movies.csv")

print("Ratings loaded:", len(ratings))
print("Movies loaded:", len(movies))

# 2. Calculate movie popularity
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# 3. Calculate weighted movie score
C = movie_stats["average_rating"].mean()
m = 5

movie_stats["weighted_score"] = (
    (movie_stats["rating_count"] /
     (movie_stats["rating_count"] + m))
    * movie_stats["average_rating"]
    +
    (m / (movie_stats["rating_count"] + m)) * C
)

# 4. Combine movie information
movie_data = movies.merge(
    movie_stats,
    on="movieId",
    how="inner"
)

# 5. Popularity-based recommendations
def popular_recommendations(number=10):

    result = movie_data.sort_values(
        "weighted_score",
        ascending=False
    ).head(number)

    return result[[
        "title",
        "genres",
        "average_rating",
        "rating_count"
    ]]


# 6. Genre-based recommendations for a new user
def genre_recommendations(genre, number=10):

    result = movie_data[
        movie_data["genres"].str.contains(
            genre,
            case=False,
            na=False
        )
    ]

    result = result.sort_values(
        "weighted_score",
        ascending=False
    )

    return result[[
        "title",
        "genres",
        "average_rating",
        "rating_count"
    ]].head(number)


# 7. Display popular recommendations
print("\nTOP 10 POPULAR MOVIES")

print(popular_recommendations(10).to_string(index=False))

# 8. Ask for a preferred genre
preferred_genre = input(
    "\nEnter your preferred genre "
    "(Action, Comedy, Drama, Romance, etc.): "
).strip()

if preferred_genre:

    print("\nRECOMMENDATIONS FOR:", preferred_genre)

    recommendations = genre_recommendations(
        preferred_genre,
        10
    )

    if recommendations.empty:
        print("No movies found for this genre.")
    else:
        print(recommendations.to_string(index=False))

# 9. Save recommendations
os.makedirs("reports", exist_ok=True)

popular_recommendations(10).to_csv(
    "reports/cold_start_popular.csv",
    index=False
)

print("\nPOPULAR RECOMMENDATIONS SAVED!")
print("DAY 10 COMPLETED!")
