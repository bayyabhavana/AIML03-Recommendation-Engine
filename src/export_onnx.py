
import os
import torch
import torch.nn as nn
import onnx


# 1. Define Wide & Deep model
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
            [user_vector, movie_vector], dim=1
        )

        deep_output = self.deep_network(combined)

        wide_output = (
            self.user_bias(user_ids)
            + self.movie_bias(movie_ids)
            + self.global_bias
        )

        return (deep_output + wide_output).squeeze(1)


# 2. Export model
def main():

    checkpoint_path = "models/ranking/wide_deep_model.pth"

    output_directory = "triton_model_repository/wide_deep_ranker/1"

    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(
        output_directory, "model.onnx"
    )

    print("Loading trained Wide & Deep model...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = WideDeepModel(
        checkpoint["number_of_users"],
        checkpoint["number_of_movies"],
        checkpoint["embedding_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully!")

    dummy_users = torch.tensor([0, 1], dtype=torch.long)
    dummy_movies = torch.tensor([0, 1], dtype=torch.long)

    print("Exporting model to ONNX...")

    torch.onnx.export(
        model,
        (dummy_users, dummy_movies),
        output_path,
        input_names=["user_ids", "movie_ids"],
        output_names=["scores"],
        dynamic_axes={
            "user_ids": {0: "batch_size"},
            "movie_ids": {0: "batch_size"},
            "scores": {0: "batch_size"}
        },
        opset_version=18,
        dynamo=False,
        external_data=False
    )

    print("Checking ONNX model...")

    onnx_model = onnx.load(output_path)

    onnx.checker.check_model(onnx_model)

    print("\nONNX EXPORT SUCCESSFUL!")
    print("Model saved at:", output_path)
    print("Model size:", round(os.path.getsize(output_path) / 1024, 2), "KB")


if __name__ == "__main__":
    main()
