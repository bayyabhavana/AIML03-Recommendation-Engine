
import os
import json
import sqlite3
import hashlib
import uuid
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


print("LOADING RECOMMENDATION SYSTEM")

# ==================================================
# 1. LOAD USER AND MOVIE MAPPINGS
# ==================================================

with open("models/retrieval/user_mapping.json", "r") as f:
    user_map = json.load(f)

with open("models/retrieval/movie_mapping.json", "r") as f:
    movie_map = json.load(f)

index_to_movie = {
    int(index): int(movie_id)
    for movie_id, index in movie_map.items()
}

index_to_user = {
    int(index): int(user_id)
    for user_id, index in user_map.items()
}

# ==================================================
# 2. LOAD MOVIE INFORMATION
# ==================================================

movies = pd.read_csv("data/processed/clean_movies.csv")

# ==================================================
# 3. DEFINE TWO-TOWER MODEL
# ==================================================

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

        similarity = (user_vector * movie_vector).sum(dim=1)

        user_bias = self.user_bias(user_ids).squeeze(-1)
        movie_bias = self.movie_bias(movie_ids).squeeze(-1)

        return similarity + user_bias + movie_bias


# ==================================================
# 4. LOAD TRAINED TWO-TOWER MODEL
# ==================================================

checkpoint = torch.load(
    "models/retrieval/two_tower_model.pth",
    map_location="cpu",
    weights_only=False
)

model = TwoTowerModel(
    checkpoint["number_of_users"],
    checkpoint["number_of_movies"],
    checkpoint["embedding_size"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ==================================================
# 5. LOAD SEEN MOVIES
# ==================================================

train = pd.read_csv("data/processed/train.csv")

seen_movies = train.groupby("userId")["movieId"].apply(
    lambda values: set(values.astype(int))
).to_dict()

# ==================================================
# 6. TRITON CONFIGURATION
# ==================================================

TRITON_URL = os.getenv(
    "TRITON_URL",
    "http://localhost:8001"
)

TRITON_MODEL = "wide_deep_ranker"

# ==================================================
# 7. CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Personalized Movie Recommendation API",
    description=(
        "Two-Tower Retrieval, Triton Ranking "
        "and Online A/B Testing"
    ),
    version="3.0"
)


@app.get("/")
def home():

    return {
        "message": "Movie Recommendation API is running!",
        "models": [
            "Two-Tower",
            "Triton Wide & Deep"
        ],
        "features": [
            "Movie recommendations",
            "Triton inference",
            "Online A/B testing",
            "Click tracking",
            "CTR reporting"
        ],
        "triton_url": TRITON_URL
    }


# ==================================================
# 8. TWO-TOWER RECOMMENDATIONS
# ==================================================

@app.get("/recommend/{user_id}")
def recommend_movies(user_id: int, top_n: int = 10):

    user_key = str(user_id)

    if user_key not in user_map:
        raise HTTPException(
            status_code=404,
            detail="User not found in training data"
        )

    if top_n < 1 or top_n > 50:
        raise HTTPException(
            status_code=400,
            detail="top_n must be between 1 and 50"
        )

    user_index = int(user_map[user_key])

    movie_indices = list(index_to_movie.keys())

    user_tensor = torch.tensor(
        [user_index] * len(movie_indices),
        dtype=torch.long
    )

    movie_tensor = torch.tensor(
        movie_indices,
        dtype=torch.long
    )

    with torch.no_grad():
        scores = model(user_tensor, movie_tensor)

    score_values = scores.numpy()

    user_seen = seen_movies.get(user_id, set())

    results = []

    for index, score in zip(movie_indices, score_values):

        movie_id = index_to_movie[index]

        if movie_id not in user_seen:
            results.append((movie_id, float(score)))

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = []

    for movie_id, score in results[:top_n]:

        movie_info = movies[movies["movieId"] == movie_id]

        if not movie_info.empty:

            recommendations.append({
                "movieId": movie_id,
                "title": movie_info.iloc[0]["title"],
                "genres": movie_info.iloc[0]["genres"],
                "score": round(score, 4)
            })

    return {
        "user_id": user_id,
        "model": "Two-Tower",
        "recommendations": recommendations
    }


# ==================================================
# 9. TRITON WIDE & DEEP PREDICTIONS
# ==================================================

@app.post("/triton/predict")
def triton_predict(user_id: int, movie_ids: list[int]):

    user_key = str(user_id)

    if user_key not in user_map:
        raise HTTPException(
            status_code=404,
            detail="User not found in training data"
        )

    if not movie_ids or len(movie_ids) > 500:
        raise HTTPException(
            status_code=400,
            detail="Provide between 1 and 500 movie IDs"
        )

    user_index = int(user_map[user_key])

    movie_indices = []

    for movie_id in movie_ids:

        movie_key = str(movie_id)

        if movie_key not in movie_map:
            raise HTTPException(
                status_code=404,
                detail=f"Movie {movie_id} not found in training data"
            )

        movie_indices.append(int(movie_map[movie_key]))

    payload = {
        "inputs": [
            {
                "name": "user_ids",
                "shape": [len(movie_indices)],
                "datatype": "INT64",
                "data": [user_index] * len(movie_indices)
            },
            {
                "name": "movie_ids",
                "shape": [len(movie_indices)],
                "datatype": "INT64",
                "data": movie_indices
            }
        ],
        "outputs": [
            {"name": "scores"}
        ]
    }

    try:

        response = requests.post(
            f"{TRITON_URL}/v2/models/{TRITON_MODEL}/infer",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        prediction_data = response.json()
        scores = prediction_data["outputs"][0]["data"]

    except requests.RequestException as error:

        raise HTTPException(
            status_code=503,
            detail=f"Triton server unavailable: {str(error)}"
        )

    recommendations = []

    for movie_id, score in zip(movie_ids, scores):

        movie_info = movies[movies["movieId"] == movie_id]

        if not movie_info.empty:

            recommendations.append({
                "movieId": movie_id,
                "title": movie_info.iloc[0]["title"],
                "genres": movie_info.iloc[0]["genres"],
                "predicted_rating": round(float(score), 4)
            })

    recommendations.sort(
        key=lambda item: item["predicted_rating"],
        reverse=True
    )

    return {
        "user_id": user_id,
        "model": "Triton Wide & Deep",
        "recommendations": recommendations
    }


# ==================================================
# 10. TRITON HEALTH CHECK
# ==================================================

@app.get("/triton/health")
def triton_health():

    try:

        response = requests.get(
            f"{TRITON_URL}/v2/health/ready",
            timeout=5
        )

        return {
            "triton_status": (
                "healthy"
                if response.status_code == 200
                else "not_ready"
            ),
            "http_status": response.status_code
        }

    except requests.RequestException as error:

        return {
            "triton_status": "unavailable",
            "error": str(error)
        }


# ==================================================
# 11. ONLINE A/B TESTING DATABASE
# ==================================================

AB_DB = "reports/online_ab_events.db"

os.makedirs("reports", exist_ok=True)


def initialize_ab_database():

    with sqlite3.connect(AB_DB) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS impressions (
                impression_id TEXT,
                user_id INTEGER,
                variant TEXT,
                movie_id INTEGER,
                position INTEGER,
                shown_at TEXT,
                clicked INTEGER DEFAULT 0
            )
        """)

        conn.commit()


initialize_ab_database()


# ==================================================
# 12. EXPERIMENT ASSIGNMENT
# ==================================================

def assign_experiment(user_id: int):

    value = int(
        hashlib.sha256(
            str(user_id).encode()
        ).hexdigest(),
        16
    )

    return "A" if value % 2 == 0 else "B"


# ==================================================
# 13. SAVE RECOMMENDATION IMPRESSIONS
# ==================================================

def save_impressions(
    user_id,
    variant,
    recommendations
):

    impression_id = str(uuid.uuid4())

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(AB_DB) as conn:

        for position, item in enumerate(
            recommendations,
            start=1
        ):

            conn.execute("""
                INSERT INTO impressions (
                    impression_id,
                    user_id,
                    variant,
                    movie_id,
                    position,
                    shown_at,
                    clicked
                )
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (
                impression_id,
                user_id,
                variant,
                int(item["movieId"]),
                position,
                timestamp
            ))

        conn.commit()

    return impression_id


# ==================================================
# 14. ONLINE A/B RECOMMENDATION ENDPOINT
# ==================================================

@app.get("/ab/recommend/{user_id}")
def online_ab_recommend(
    user_id: int,
    top_n: int = 10
):

    if top_n < 1 or top_n > 10:
        raise HTTPException(
            status_code=400,
            detail="top_n must be between 1 and 10"
        )

    if str(user_id) not in user_map:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    variant = assign_experiment(user_id)

    if variant == "A":

        # Experiment A: Two-Tower
        result = recommend_movies(user_id, top_n)

        recommendations = result["recommendations"]

        model_name = "Two-Tower"

    else:

        # Experiment B:
        # Two-Tower candidate retrieval,
        # followed by Triton Wide & Deep ranking.

        candidates = recommend_movies(user_id, 50)

        candidate_ids = [
            item["movieId"]
            for item in candidates["recommendations"]
        ]

        if candidate_ids:

            ranked = triton_predict(
                user_id,
                candidate_ids
            )

            recommendations = (
                ranked["recommendations"][:top_n]
            )

        else:
            recommendations = []

        model_name = "Two-Tower + Triton Wide & Deep"

    impression_id = save_impressions(
        user_id,
        variant,
        recommendations
    )

    return {
        "experiment": "Online A/B Test",
        "variant": variant,
        "model": model_name,
        "user_id": user_id,
        "impression_id": impression_id,
        "recommendations": recommendations,
        "message": "Record actual clicks using POST /ab/click"
    }


# ==================================================
# 15. CLICK EVENT REQUEST FORMAT
# ==================================================

class ClickEvent(BaseModel):

    impression_id: str
    movie_id: int


# ==================================================
# 16. RECORD ACTUAL USER CLICKS
# ==================================================

@app.post("/ab/click")
def record_ab_click(event: ClickEvent):

    with sqlite3.connect(AB_DB) as conn:

        cursor = conn.execute("""
            UPDATE impressions
            SET clicked = 1
            WHERE impression_id = ?
              AND movie_id = ?
              AND clicked = 0
        """, (
            event.impression_id,
            event.movie_id
        ))

        conn.commit()

        if cursor.rowcount == 0:

            raise HTTPException(
                status_code=404,
                detail="Impression not found or already clicked"
            )

    return {
        "status": "success",
        "message": "Click recorded successfully",
        "movie_id": event.movie_id
    }


# ==================================================
# 17. ONLINE A/B TESTING CTR REPORT
# ==================================================

@app.get("/ab/report")
def online_ab_report():

    with sqlite3.connect(AB_DB) as conn:

        rows = conn.execute("""
            SELECT
                variant,
                COUNT(*) AS impressions,
                SUM(clicked) AS clicks
            FROM impressions
            GROUP BY variant
            ORDER BY variant
        """).fetchall()

    report = []

    for variant, impressions, clicks in rows:

        clicks = clicks or 0

        ctr = (
            clicks / impressions * 100
            if impressions > 0
            else 0
        )

        report.append({
            "variant": variant,
            "model": (
                "Two-Tower"
                if variant == "A"
                else "Two-Tower + Triton Wide & Deep"
            ),
            "impressions": impressions,
            "clicks": clicks,
            "ctr_percent": round(ctr, 2)
        })

    return {
        "experiment": "Online A/B Testing",
        "results": report
    }


# ==================================================
# 18. API STARTUP MESSAGE
# ==================================================

print("RECOMMENDATION API READY")
print("ONLINE A/B TESTING READY")
print("CTR TRACKING READY")
