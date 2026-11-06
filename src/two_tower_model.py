
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("STARTING TWO-TOWER RECOMMENDATION MODEL")

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# 2. Load training data
ratings = pd.read_csv("data/processed/train.csv")

print("Total training ratings:", len(ratings))

# 3. Select positive ratings
# Ratings of 4 or above are treated as positive preferences
positive_ratings = ratings[ratings["rating"] >= 4.0].copy()

print("Positive interactions:", len(positive_ratings))

# 4. Create user and movie ID mappings
user_ids = positive_ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {
    int(user_id): index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    int(movie_id): index
    for index, movie_id in enumerate(movie_ids)
}

positive_ratings = positive_ratings[
    positive_ratings["movieId"].isin(movie_to_index)
]

user_indices = positive_ratings["userId"].map(user_to_index).to_numpy()
movie_indices = positive_ratings["movieId"].map(movie_to_index).to_numpy()

# 5. Create PyTorch dataset
users = torch.tensor(user_indices, dtype=torch.long)
movies = torch.tensor(movie_indices, dtype=torch.long)

dataset = TensorDataset(users, movies)

loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True
)

# 6. Build Two-Tower model
class TwoTowerModel(nn.Module):

    def __init__(self, number_of_users, number_of_movies,
                 embedding_size=32):

        super().__init__()

        # User tower
        self.user_embedding = nn.Embedding(
            number_of_users, embedding_size
        )

        # Movie tower
        self.movie_embedding = nn.Embedding(
            number_of_movies, embedding_size
        )

        self.user_bias = nn.Embedding(number_of_users, 1)
        self.movie_bias = nn.Embedding(number_of_movies, 1)

    def forward(self, user_ids, movie_ids):

        user_vector = self.user_embedding(user_ids)
        movie_vector = self.movie_embedding(movie_ids)

        similarity = (user_vector * movie_vector).sum(dim=1)

        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()

        return similarity + user_bias + movie_bias


# 7. Initialize model
model = TwoTowerModel(
    number_of_users=len(user_to_index),
    number_of_movies=len(movie_to_index),
    embedding_size=32
).to(device)

# 8. Set loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 9. Train model using negative sampling
print("\nSTARTING MODEL TRAINING")

number_of_epochs = 5
number_of_movies = len(movie_to_index)

for epoch in range(number_of_epochs):

    model.train()
    total_loss = 0

    for batch_users, positive_movies in loader:

        batch_users = batch_users.to(device)
        positive_movies = positive_movies.to(device)

        # Randomly select negative movies
        negative_movies = torch.randint(
            0,
            number_of_movies,
            positive_movies.shape,
            device=device
        )

        # Positive interaction scores
        positive_scores = model(
            batch_users, positive_movies
        )

        # Negative interaction scores
        negative_scores = model(
            batch_users, negative_movies
        )

        # Positive labels = 1
        positive_labels = torch.ones_like(positive_scores)

        # Negative labels = 0
        negative_labels = torch.zeros_like(negative_scores)

        positive_loss = criterion(
            positive_scores, positive_labels
        )

        negative_loss = criterion(
            negative_scores, negative_labels
        )

        loss = positive_loss + negative_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{number_of_epochs} "
        f"- Loss: {average_loss:.4f}"
    )

# 10. Save trained model
os.makedirs("models/retrieval", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "number_of_users": len(user_to_index),
        "number_of_movies": len(movie_to_index),
        "embedding_size": 32
    },
    "models/retrieval/two_tower_model.pth"
)

# Save ID mappings
with open("models/retrieval/user_mapping.json", "w") as file:
    json.dump(user_to_index, file)

with open("models/retrieval/movie_mapping.json", "w") as file:
    json.dump(movie_to_index, file)

print("\nMODEL SAVED SUCCESSFULLY!")
print("TWO-TOWER TRAINING COMPLETED!")
