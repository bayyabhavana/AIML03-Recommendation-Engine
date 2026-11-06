
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

print("STARTING MODEL EVALUATION")

# 1. Select CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluation device:", device)

# 2. Load datasets
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("Training records:", len(train))
print("Testing records:", len(test))

# 3. Load saved ID mappings
with open("models/retrieval/user_mapping.json", "r") as file:
    user_to_index = json.load(file)

with open("models/retrieval/movie_mapping.json", "r") as file:
    movie_to_index = json.load(file)

# Convert mapping keys to integers
user_to_index = {
    int(key): value for key, value in user_to_index.items()
}

movie_to_index = {
    int(key): value for key, value in movie_to_index.items()
}

index_to_movie = {
    value: key for key, value in movie_to_index.items()
}

# 4. Define the same Two-Tower architecture
class TwoTowerModel(nn.Module):

    def __init__(self, number_of_users, number_of_movies,
                 embedding_size=32):

        super().__init__()

        self.user_embedding = nn.Embedding(
            number_of_users, embedding_size
        )

        self.movie_embedding = nn.Embedding(
            number_of_movies, embedding_size
        )

        self.user_bias = nn.Embedding(number_of_users, 1)
        self.movie_bias = nn.Embedding(number_of_movies, 1)

    def forward(self, user_ids, movie_ids):

        user_vector = self.user_embedding(user_ids)
        movie_vector = self.movie_embedding(movie_ids)

        similarity = (user_vector * movie_vector).sum(dim=1)

        user_bias = self.user_bias(user_ids).squeeze(-1)
        movie_bias = self.movie_bias(movie_ids).squeeze(-1)

        return similarity + user_bias + movie_bias


# 5. Load trained model
checkpoint = torch.load(
    "models/retrieval/two_tower_model.pth",
    map_location=device,
    weights_only=True
)

model = TwoTowerModel(
    number_of_users=checkpoint["number_of_users"],
    number_of_movies=checkpoint["number_of_movies"],
    embedding_size=checkpoint["embedding_size"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nTRAINED MODEL LOADED SUCCESSFULLY!")

# 6. Prepare test data
# Only ratings of 4 or above are considered relevant
test_positive = test[test["rating"] >= 4.0].copy()

# Only evaluate users and movies known to the trained model
test_positive = test_positive[
    test_positive["userId"].isin(user_to_index) &
    test_positive["movieId"].isin(movie_to_index)
]

# Group relevant movies by user
relevant_by_user = (
    test_positive.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Exclude movies already rated by each user in training
seen_by_user = (
    train.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# All movies available in the trained movie catalogue
all_movie_indices = np.arange(
    len(movie_to_index),
    dtype=np.int64
)

# 7. Evaluation metrics
K = 10

precision_scores = []
recall_scores = []
ndcg_scores = []

print("\nCALCULATING PRECISION@10, RECALL@10 AND NDCG@10")

# 8. Evaluate each user
with torch.no_grad():

    for user_id, relevant_movies in relevant_by_user.items():

        if user_id not in user_to_index:
            continue

        # Convert relevant movie IDs to model indices
        relevant_indices = {
            movie_to_index[movie_id]
            for movie_id in relevant_movies
        }

        # Remove movies already seen during training
        seen_movies = seen_by_user.get(user_id, set())

        seen_indices = {
            movie_to_index[movie_id]
            for movie_id in seen_movies
            if movie_id in movie_to_index
        }

        candidate_indices = np.array(
            [
                movie_index
                for movie_index in all_movie_indices
                if movie_index not in seen_indices
            ],
            dtype=np.int64
        )

        if len(candidate_indices) == 0:
            continue

        user_index = user_to_index[user_id]

        # Score all unseen candidate movies
        user_tensor = torch.full(
            (len(candidate_indices),),
            user_index,
            dtype=torch.long,
            device=device
        )

        movie_tensor = torch.tensor(
            candidate_indices,
            dtype=torch.long,
            device=device
        )

        scores = model(user_tensor, movie_tensor)

        # Select top K recommendations
        top_count = min(K, len(candidate_indices))

        top_positions = torch.topk(
            scores,
            k=top_count
        ).indices.cpu().numpy()

        recommended_indices = candidate_indices[top_positions]

        # Check which recommendations are relevant
        hits = [
            1 if movie_index in relevant_indices else 0
            for movie_index in recommended_indices
        ]

        number_of_hits = sum(hits)

        # Precision@K
        precision = number_of_hits / K

        # Recall@K
        recall = number_of_hits / len(relevant_indices)

        # NDCG@K
        dcg = sum(
            hit / np.log2(position + 2)
            for position, hit in enumerate(hits)
        )

        ideal_hits = min(len(relevant_indices), K)

        idcg = sum(
            1 / np.log2(position + 2)
            for position in range(ideal_hits)
        )

        ndcg = dcg / idcg if idcg > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

# 9. Display final results
print("\n========== EVALUATION RESULTS ==========")

if len(precision_scores) == 0:
    print("No eligible users found for evaluation.")
else:

    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_ndcg = np.mean(ndcg_scores)

    print("Evaluated users:", len(precision_scores))
    print(f"Precision@10: {mean_precision:.4f}")
    print(f"Recall@10:    {mean_recall:.4f}")
    print(f"NDCG@10:      {mean_ndcg:.4f}")

    # 10. Save results
    os.makedirs("reports", exist_ok=True)

    results = pd.DataFrame({
        "metric": [
            "Precision@10",
            "Recall@10",
            "NDCG@10"
        ],
        "value": [
            mean_precision,
            mean_recall,
            mean_ndcg
        ]
    })

    results.to_csv(
        "reports/evaluation_results.csv",
        index=False
    )

    print("\nRESULTS SAVED TO reports/evaluation_results.csv")
    print("\nDAY 7 EVALUATION COMPLETED!")
