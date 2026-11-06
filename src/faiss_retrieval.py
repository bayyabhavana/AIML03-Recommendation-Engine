
import os
import json
import numpy as np
import torch
import faiss
import torch.nn as nn

print("STARTING FAISS MOVIE RETRIEVAL")

# 1. Set device
device = torch.device("cpu")

# 2. Load trained model checkpoint
checkpoint = torch.load(
    "models/retrieval/two_tower_model.pth",
    map_location=device,
    weights_only=True
)

number_of_users = checkpoint["number_of_users"]
number_of_movies = checkpoint["number_of_movies"]
embedding_size = checkpoint["embedding_size"]

# 3. Recreate the Two-Tower model
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


# 4. Load model weights
model = TwoTowerModel(
    number_of_users,
    number_of_movies,
    embedding_size
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("TRAINED MODEL LOADED SUCCESSFULLY")

# 5. Load movie ID mapping
with open("models/retrieval/movie_mapping.json", "r") as file:
    movie_to_index = json.load(file)

# Sort by embedding index so IDs match the model embedding rows
movie_ids = np.array(
    [
        int(movie_id)
        for movie_id, index in sorted(
            movie_to_index.items(),
            key=lambda item: item[1]
        )
    ],
    dtype=np.int64
)

# 6. Extract movie embeddings
with torch.no_grad():

    movie_embeddings = model.movie_embedding.weight.cpu().numpy()

# Convert to FAISS-compatible format
movie_embeddings = np.asarray(
    movie_embeddings,
    dtype=np.float32
)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(movie_embeddings)

print("MOVIE EMBEDDINGS EXTRACTED:", movie_embeddings.shape)

# 7. Create FAISS index
dimension = movie_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# Add movie vectors to index
index.add(movie_embeddings)

print("FAISS INDEX CREATED SUCCESSFULLY")
print("TOTAL MOVIES INDEXED:", index.ntotal)

# 8. Save FAISS index and movie IDs
os.makedirs("models/faiss", exist_ok=True)

faiss.write_index(
    index,
    "models/faiss/movie_index.faiss"
)

np.save(
    "models/faiss/movie_ids.npy",
    movie_ids
)

print("FAISS INDEX SAVED SUCCESSFULLY")

# 9. Test a similarity search
query_vector = movie_embeddings[0:1].copy()

scores, positions = index.search(query_vector, 5)

print("\nSIMILAR MOVIE SEARCH TEST")

for score, position in zip(scores[0], positions[0]):

    if position >= 0:
        print(
            "Movie ID:",
            movie_ids[position],
            "| Similarity:",
            round(float(score), 4)
        )

print("\nDAY 8 FAISS RETRIEVAL COMPLETED!")

