
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, min, max
import os

print("STARTING APACHE SPARK PROCESSING")

# 1. Start Spark
spark = SparkSession.builder \
    .appName("MovieRecommendationEngine") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("SPARK STARTED SUCCESSFULLY")

# 2. Load training dataset
ratings = spark.read.csv(
    "data/processed/train.csv",
    header=True,
    inferSchema=True
)

print("\nTRAINING DATA")
ratings.show(5)

# 3. Count total ratings
total_ratings = ratings.count()

print("\nTOTAL RATINGS:")
print(total_ratings)

# 4. Generate user features
user_features = ratings.groupBy("userId").agg(
    count("rating").alias("total_ratings"),
    avg("rating").alias("average_rating")
)

print("\nUSER FEATURES")
user_features.show(5)

# 5. Generate movie features
movie_features = ratings.groupBy("movieId").agg(
    count("rating").alias("total_ratings"),
    avg("rating").alias("average_rating"),
    min("rating").alias("minimum_rating"),
    max("rating").alias("maximum_rating")
)

print("\nMOVIE FEATURES")
movie_features.show(5)

# 6. Create output directory
output_folder = "data/processed/spark"
os.makedirs(output_folder, exist_ok=True)

# 7. Save user features as CSV
user_features.toPandas().to_csv(
    os.path.join(output_folder, "user_features.csv"),
    index=False
)

print("\nUSER FEATURES SAVED SUCCESSFULLY!")

# 8. Save movie features as CSV
movie_features.toPandas().to_csv(
    os.path.join(output_folder, "movie_features.csv"),
    index=False
)

print("MOVIE FEATURES SAVED SUCCESSFULLY!")

# 9. Final message
print("\nSPARK PROCESSING COMPLETED SUCCESSFULLY!")

# 10. Stop Spark
spark.stop()
