
import redis
import json
import pandas as pd

print("STARTING REDIS CACHE SYSTEM")

# Connect to Redis
cache = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

# Test connection
if cache.ping():
    print("REDIS CONNECTED SUCCESSFULLY!")

# Load popular movie recommendations
movies = pd.read_csv("reports/cold_start_popular.csv")

recommendations = movies.to_dict(orient="records")

# Store recommendations in Redis
cache.set(
    "popular_movies",
    json.dumps(recommendations),
    ex=3600
)

print("\nPOPULAR MOVIES STORED IN REDIS!")

# Retrieve recommendations from Redis
cached_data = cache.get("popular_movies")

if cached_data:
    result = json.loads(cached_data)

    print("\nRETRIEVED RECOMMENDATIONS:")

    for movie in result[:10]:
        print(
            movie["title"],
            "| Rating:",
            round(movie["average_rating"], 2)
        )

    print("\nCACHE HIT SUCCESSFUL!")

else:
    print("NO DATA FOUND IN CACHE")

print("\nDAY 11 COMPLETED!")
