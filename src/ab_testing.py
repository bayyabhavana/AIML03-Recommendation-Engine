
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

print("STARTING OFFLINE A/B MODEL COMPARISON")

# --------------------------------------------------
# 1. DEVICE
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluation device:", device)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("Training records:", len(train))
print("Testing records:", len(test))

# --------------------------------------------------
# 3. DEFINE TWO-TOWER MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# 4. DEFINE WIDE & DEEP MODEL
# --------------------------------------------------

class WideDeepModel(nn.Module):

    def __init__(self, number_of_users,
                 number_of_movies, embedding_size=32):

        super().__init__()

        self.user_embedding = nn.Embedding(
            number_of_users, embedding_size
        )

        self.movie_embedding = nn.Embedding(
            number_of_movies, embedding_size
        )

        self.deep_network = nn.Sequential(
            nn.Linear(embedding_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.user_bias = nn.Embedding(number_of_users, 1)
        self.movie_bias = nn.Embedding(number_of_movies, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, movie_ids):

        user_vector = self.user_embedding(user_ids)
        movie_vector = self.movie_embedding(movie_ids)

        combined = torch.cat(
            [user_vector, movie_vector], dim=1
        )

        deep_output = self.deep_network(combined)

        wide_output = (
            self.user_bias(user_ids)
            + self.movie_bias(movie_ids)
            + self.global_bias
        )

        return (deep_output + wide_output).squeeze(1)


# --------------------------------------------------
# 5. LOAD TWO-TOWER CHECKPOINT AND MAPPINGS
# --------------------------------------------------

two_checkpoint = torch.load(
    "models/retrieval/two_tower_model.pth",
    map_location=device,
    weights_only=True
)

with open("models/retrieval/user_mapping.json", "r") as file:
    two_user_map = json.load(file)

with open("models/retrieval/movie_mapping.json", "r") as file:
    two_movie_map = json.load(file)

two_user_map = {
    int(key): int(value)
    for key, value in two_user_map.items()
}

two_movie_map = {
    int(key): int(value)
    for key, value in two_movie_map.items()
}

two_model = TwoTowerModel(
    two_checkpoint["number_of_users"],
    two_checkpoint["number_of_movies"],
    two_checkpoint["embedding_size"]
).to(device)

two_model.load_state_dict(two_checkpoint["model_state_dict"])
two_model.eval()

print("\nEXPERIMENT A: TWO-TOWER MODEL LOADED")


# --------------------------------------------------
# 6. LOAD WIDE & DEEP CHECKPOINT AND MAPPINGS
# --------------------------------------------------

wide_checkpoint = torch.load(
    "models/ranking/wide_deep_model.pth",
    map_location=device,
    weights_only=True
)

wide_user_map = wide_checkpoint["user_map"]
wide_movie_map = wide_checkpoint["movie_map"]

wide_model = WideDeepModel(
    wide_checkpoint["number_of_users"],
    wide_checkpoint["number_of_movies"],
    wide_checkpoint["embedding_size"]
).to(device)

wide_model.load_state_dict(wide_checkpoint["model_state_dict"])
wide_model.eval()

print("EXPERIMENT B: WIDE & DEEP MODEL LOADED")


# --------------------------------------------------
# 7. PREPARE COMMON USERS AND MOVIES
# --------------------------------------------------

common_users = (
    set(two_user_map.keys())
    & set(wide_user_map.keys())
)

common_movies = (
    set(two_movie_map.keys())
    & set(wide_movie_map.keys())
)

print("\nCommon users:", len(common_users))
print("Common movies:", len(common_movies))

# Only ratings >= 4 are treated as relevant
test_positive = test[
    (test["rating"] >= 4.0)
    & test["userId"].isin(common_users)
    & test["movieId"].isin(common_movies)
].copy()

relevant_by_user = (
    test_positive.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

seen_by_user = (
    train.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)


# --------------------------------------------------
# 8. EVALUATION FUNCTION
# --------------------------------------------------

K = 10

def calculate_metrics(model, user_map, movie_map):

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id, relevant_movies in relevant_by_user.items():

        if user_id not in user_map:
            continue

        # Exclude movies already rated during training
        seen_movies = seen_by_user.get(user_id, set())

        candidates = [
            movie_id
            for movie_id in common_movies
            if movie_id not in seen_movies
            and movie_id in movie_map
        ]

        if len(candidates) < K:
            continue

        relevant = relevant_movies.intersection(candidates)

        if len(relevant) == 0:
            continue

        user_index = user_map[user_id]

        user_tensor = torch.full(
            (len(candidates),),
            user_index,
            dtype=torch.long,
            device=device
        )

        movie_tensor = torch.tensor(
            [movie_map[movie_id] for movie_id in candidates],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            scores = model(user_tensor, movie_tensor)

        top_positions = torch.topk(
            scores,
            k=K
        ).indices.cpu().numpy()

        recommended_movies = [
            candidates[position]
            for position in top_positions
        ]

        hits = [
            1 if movie_id in relevant else 0
            for movie_id in recommended_movies
        ]

        number_of_hits = sum(hits)

        precision = number_of_hits / K
        recall = number_of_hits / len(relevant)

        dcg = sum(
            hit / np.log2(position + 2)
            for position, hit in enumerate(hits)
        )

        ideal_hits = min(len(relevant), K)

        idcg = sum(
            1 / np.log2(position + 2)
            for position in range(ideal_hits)
        )

        ndcg = dcg / idcg if idcg > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

    if len(precision_scores) == 0:
        raise ValueError(
            "No eligible users found for model evaluation."
        )

    return {
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
        "ndcg": float(np.mean(ndcg_scores)),
        "evaluated_users": len(precision_scores)
    }


# --------------------------------------------------
# 9. EVALUATE BOTH MODELS
# --------------------------------------------------

print("\nEVALUATING EXPERIMENT A: TWO-TOWER")

result_a = calculate_metrics(
    two_model,
    two_user_map,
    two_movie_map
)

print("\nEVALUATING EXPERIMENT B: WIDE & DEEP")

result_b = calculate_metrics(
    wide_model,
    wide_user_map,
    wide_movie_map
)


# --------------------------------------------------
# 10. CREATE COMPARISON REPORT
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "experiment": "A",
        "model": "Two-Tower",
        **result_a
    },
    {
        "experiment": "B",
        "model": "Wide & Deep",
        **result_b
    }
])

os.makedirs("reports", exist_ok=True)

comparison.to_csv(
    "reports/ab_testing_results.csv",
    index=False
)

print("\n========== A/B TESTING RESULTS ==========")
print(comparison.to_string(index=False))

print("\n========== MODEL COMPARISON ==========")

if result_a["ndcg"] > result_b["ndcg"]:
    print("Two-Tower performed better on NDCG@10.")
elif result_b["ndcg"] > result_a["ndcg"]:
    print("Wide & Deep performed better on NDCG@10.")
else:
    print("Both models achieved equal NDCG@10.")

print("\nREPORT SAVED:")
print("reports/ab_testing_results.csv")

print("\nOFFLINE A/B MODEL COMPARISON COMPLETED!")
