
import pandas as pd
import os

print("STARTING TIME-BASED DATA SPLITTING")

# Step 1: Load cleaned ratings
ratings = pd.read_csv(
    "data/processed/clean_ratings.csv"
)

# Step 2: Convert timestamp into date
ratings["timestamp"] = pd.to_datetime(
    ratings["timestamp"],
    unit="s",
    utc=True
)

# Step 3: Sort records by time
ratings = ratings.sort_values(
    "timestamp"
)

# Step 4: Find 80% cutoff date
split_date = ratings["timestamp"].quantile(0.8)

# Step 5: Split training and testing data
train_data = ratings[
    ratings["timestamp"] <= split_date
].copy()

test_data = ratings[
    ratings["timestamp"] > split_date
].copy()

# Step 6: Create output folder
os.makedirs("data/processed", exist_ok=True)

# Step 7: Save training data
train_data.to_csv(
    "data/processed/train.csv",
    index=False
)

# Step 8: Save testing data
test_data.to_csv(
    "data/processed/test.csv",
    index=False
)

# Step 9: Display results
print("\nSplit date:", split_date)

print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining data sample:")
print(train_data.head())

print("\nTesting data sample:")
print(test_data.head())

print("\nTIME-BASED SPLITTING COMPLETED!")
