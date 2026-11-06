
import pandas as pd
import os

print("WELCOME TO MY RECOMMENDATION ENGINE")

# Step 1: Dataset location
ratings_file = "data/ml-25m/ratings.csv"
movies_file = "data/ml-25m/movies.csv"

# Step 2: Check dataset files
if os.path.exists(ratings_file) and os.path.exists(movies_file):

    print("Dataset found successfully!")

    # Step 3: Read movie data
    movies = pd.read_csv(movies_file)

    # Step 4: Read ratings data
    ratings = pd.read_csv(ratings_file, nrows=10)

    # Step 5: Display movies
    print("\nMOVIE DETAILS")
    print(movies.head())

    # Step 6: Display ratings
    print("\nUSER RATINGS")
    print(ratings)

    # Step 7: Display dataset information
    print("\nMovie columns:")
    print(movies.columns)

    print("\nRating columns:")
    print(ratings.columns)

else:
    print("Dataset not found!")
    print("Please check your folder location.")

print("\nDAY 1 COMPLETED!")
