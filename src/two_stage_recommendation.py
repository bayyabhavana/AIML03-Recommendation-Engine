
import os
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import faiss

print("=" * 55)
print("PERSONALIZED TWO-STAGE RECOMMENDATION ENGINE")
print("=" * 55)

# --------------------------------------------------
# 1. SELECT DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


# --------------------------------------------------
# 2. GET USER ID
# --------------------------------------------------

user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 173

print("Selected User ID:", user_id)


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

print("\nLOADING DATA")

train = pd.read_csv("data/processed/train.csv")
movies = pd.read_csv("data/processed/clean_movies.csv")

print("Training records:", len(train))
print("Movies loaded:", len(movies))


# --------------------------------------------------
# 4. DEFINE TWO-TOWER MODEL
# --------------------------------------------------

class TwoTowerModel(nn.Module):

    def __init__(
        self,
        number_of_users,
        number_of_movies,
        embedding_size=32
    ):

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

        similarity = (
            user_vector * movie_vector
        ).sum(dim=1)

        user_bias = self.user_bias(user_ids).squeeze(-1)
        movie_bias = self.movie_bias(movie_ids).squeeze(-1)

        return similarity + user_bias + movie_bias


# --------------------------------------------------
# 5. DEFINE WIDE & DEEP MODEL
# --------------------------------------------------

class WideDeepModel(nn.Module):

    def __init__(
        self,
        number_of_users,
        number_of_movies,
        embedding_size=32
    ):

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
            [user_vector, movie_vector],
            dim=1
        )

        deep_output = self.deep_network(combined)

        wide_output = (
            self.user_bias(user_ids)
            + self.movie_bias(movie_ids)
            + self.global_bias
        )

        return (deep_output + wide_output).squeeze(1)


# --------------------------------------------------
# 6. LOAD TWO-TOWER MODEL
# --------------------------------------------------

print("\nLOADING TWO-TOWER MODEL")

two_checkpoint = torch.load(
    "models/retrieval/two_tower_model.pth",
    map_location=device,
    weights_only=True
)

with open(
    "models/retrieval/user_mapping.json", "r"
) as file:

    two_user_map = json.load(file)

with open(
    "models/retrieval/movie_mapping.json", "r"
) as file:

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

two_model.load_state_dict(
    two_checkpoint["model_state_dict"]
)

two_model.eval()

print("Two-Tower model loaded successfully!")


# --------------------------------------------------
# 7. LOAD WIDE & DEEP MODEL
# --------------------------------------------------

print("\nLOADING WIDE & DEEP MODEL")

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

wide_model.load_state_dict(
    wide_checkpoint["model_state_dict"]
)

wide_model.eval()

print("Wide & Deep model loaded successfully!")


# --------------------------------------------------
# 8. CHECK USER MAPPINGS
# --------------------------------------------------

if user_id not in two_user_map:

    raise ValueError(
        f"User {user_id} is not available in Two-Tower model."
    )

if user_id not in wide_user_map:

    raise ValueError(
        f"User {user_id} is not available in Wide & Deep model."
    )

two_user_index = two_user_map[user_id]
wide_user_index = wide_user_map[user_id]


# --------------------------------------------------
# 9. LOAD FAISS INDEX
# --------------------------------------------------

print("\nLOADING FAISS INDEX")

faiss_index = faiss.read_index(
    "models/faiss/movie_index.faiss"
)

faiss_movie_ids = np.load(
    "models/faiss/movie_ids.npy"
)

if faiss_index.ntotal != len(faiss_movie_ids):

    raise ValueError(
        "FAISS index and movie ID mapping sizes do not match."
    )

print("FAISS index loaded successfully!")
print("Movies indexed:", faiss_index.ntotal)


# --------------------------------------------------
# 10. FIND UNSEEN MOVIES
# --------------------------------------------------

seen_movies = set(
    train.loc[
        train["userId"] == user_id,
        "movieId"
    ].astype(int)
)

common_movies = (
    set(two_movie_map.keys())
    & set(wide_movie_map.keys())
)

print("\nMovies already seen:", len(seen_movies))
print("Common movie catalogue:", len(common_movies))


# --------------------------------------------------
# 11. STAGE ONE: FAISS CANDIDATE RETRIEVAL
# --------------------------------------------------

print("\n" + "=" * 55)
print("STAGE 1: FAISS PERSONALIZED RETRIEVAL")
print("=" * 55)

# Get the selected user's Two-Tower embedding

with torch.no_grad():

    user_vector = (
        two_model.user_embedding.weight[
            two_user_index
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .reshape(1, -1)
    )

# Normalize user vector for cosine similarity

faiss.normalize_L2(user_vector)

# Search the complete FAISS index.
# This allows us to skip previously seen movies
# and still find up to 100 eligible candidates.

similarities, positions = faiss_index.search(
    user_vector,
    faiss_index.ntotal
)

retrieval_results = []

for similarity, position in zip(
    similarities[0],
    positions[0]
):

    if position < 0:
        continue

    movie_id = int(faiss_movie_ids[position])

    # Exclude previously seen movies
    if movie_id in seen_movies:
        continue

    # Only keep movies supported by both models
    if movie_id not in common_movies:
        continue

    retrieval_results.append({
        "movieId": movie_id,
        "retrieval_score": float(similarity)
    })

    if len(retrieval_results) == 100:
        break

if len(retrieval_results) == 0:

    raise ValueError(
        "No eligible movies found by FAISS."
    )

top_candidates = pd.DataFrame(retrieval_results)

print("FAISS retrieval completed!")
print("Candidates generated:", len(top_candidates))


# --------------------------------------------------
# 12. STAGE TWO: WIDE & DEEP RANKING
# --------------------------------------------------

print("\n" + "=" * 55)
print("STAGE 2: WIDE & DEEP RANKING")
print("=" * 55)

wide_user_tensor = torch.full(
    (len(top_candidates),),
    wide_user_index,
    dtype=torch.long,
    device=device
)

wide_movie_tensor = torch.tensor(
    [
        wide_movie_map[int(movie_id)]
        for movie_id in top_candidates["movieId"]
    ],
    dtype=torch.long,
    device=device
)

with torch.no_grad():

    ranking_scores = wide_model(
        wide_user_tensor,
        wide_movie_tensor
    ).cpu().numpy()

top_candidates["ranking_score"] = ranking_scores

# Sort by Wide & Deep ranking scores

final_recommendations = (
    top_candidates
    .sort_values(
        "ranking_score",
        ascending=False
    )
    .head(10)
    .copy()
)

print("Ranking completed!")
print("Final recommendations selected:", len(final_recommendations))


# --------------------------------------------------
# 13. ADD MOVIE TITLES AND GENRES
# --------------------------------------------------

movies["movieId"] = movies["movieId"].astype(int)

final_recommendations["movieId"] = (
    final_recommendations["movieId"].astype(int)
)

final_recommendations = final_recommendations.merge(
    movies,
    on="movieId",
    how="left"
)

display_columns = [
    column
    for column in [
        "movieId",
        "title",
        "genres",
        "retrieval_score",
        "ranking_score"
    ]
    if column in final_recommendations.columns
]

final_recommendations = final_recommendations[
    display_columns
]


# --------------------------------------------------
# 14. SAVE FINAL RECOMMENDATIONS
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

output_file = (
    f"reports/two_stage_user_{user_id}.csv"
)

final_recommendations.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 15. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL TOP 10 PERSONALIZED MOVIES")
print("=" * 55)

print(
    final_recommendations.to_string(index=False)
)

print("\nResults saved to:", output_file)

print("\n" + "=" * 55)
print("TWO-STAGE RECOMMENDATION COMPLETED!")
print("=" * 55)
