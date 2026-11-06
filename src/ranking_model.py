
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("STARTING DEEP LEARNING RANKING MODEL")

# 1. Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Training device:", device)

# 2. Load train and test data
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# 3. Create ID mappings using training data
users = train["userId"].unique()
movies = train["movieId"].unique()

user_map = {int(v): i for i, v in enumerate(users)}
movie_map = {int(v): i for i, v in enumerate(movies)}

# 4. Convert IDs into model indices
train = train[
    train["userId"].isin(user_map) &
    train["movieId"].isin(movie_map)
].copy()

test = test[
    test["userId"].isin(user_map) &
    test["movieId"].isin(movie_map)
].copy()

train["user_index"] = train["userId"].map(user_map)
train["movie_index"] = train["movieId"].map(movie_map)

test["user_index"] = test["userId"].map(user_map)
test["movie_index"] = test["movieId"].map(movie_map)

print("Training records:", len(train))
print("Testing records:", len(test))

# 5. Define Wide & Deep model
class WideDeepModel(nn.Module):

    def __init__(self, number_of_users,
                 number_of_movies, embedding_size=32):

        super().__init__()

        # Deep component
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

        # Wide component
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


# 6. Prepare training tensors
train_users = torch.tensor(
    train["user_index"].to_numpy(),
    dtype=torch.long
)

train_movies = torch.tensor(
    train["movie_index"].to_numpy(),
    dtype=torch.long
)

train_ratings = torch.tensor(
    train["rating"].to_numpy(),
    dtype=torch.float32
)

dataset = TensorDataset(
    train_users, train_movies, train_ratings
)

loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True
)

# 7. Initialize model
model = WideDeepModel(
    len(user_map),
    len(movie_map)
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# 8. Train model
print("\nSTARTING RANKING MODEL TRAINING")

for epoch in range(5):

    model.train()
    total_loss = 0

    for batch_users, batch_movies, batch_ratings in loader:

        batch_users = batch_users.to(device)
        batch_movies = batch_movies.to(device)
        batch_ratings = batch_ratings.to(device)

        predictions = model(batch_users, batch_movies)

        loss = criterion(predictions, batch_ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/5 - "
        f"Loss: {total_loss / len(loader):.4f}"
    )

# 9. Evaluate using test data
model.eval()

test_users = torch.tensor(
    test["user_index"].to_numpy(),
    dtype=torch.long,
    device=device
)

test_movies = torch.tensor(
    test["movie_index"].to_numpy(),
    dtype=torch.long,
    device=device
)

actual_ratings = test["rating"].to_numpy()

with torch.no_grad():

    predictions = model(
        test_users, test_movies
    ).cpu().numpy()

# Keep predictions within the MovieLens rating range
predictions = np.clip(predictions, 0.5, 5.0)

rmse = np.sqrt(
    np.mean((actual_ratings - predictions) ** 2)
)

mae = np.mean(
    np.abs(actual_ratings - predictions)
)

print("\nRANKING MODEL EVALUATION")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 10. Save model and mappings
os.makedirs("models/ranking", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "number_of_users": len(user_map),
        "number_of_movies": len(movie_map),
        "embedding_size": 32,
        "user_map": user_map,
        "movie_map": movie_map
    },
    "models/ranking/wide_deep_model.pth"
)

print("\nRANKING MODEL SAVED SUCCESSFULLY!")
print("DAY 9 COMPLETED!")
