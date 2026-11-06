# Large-Scale Personalized Recommendation Engine

A two-stage movie recommendation system that generates personalized movie recommendations using machine learning, vector similarity search, and deep learning-based ranking.

## Project Overview

This project implements a personalized recommendation engine using the MovieLens dataset. It combines candidate retrieval and ranking to generate relevant movie recommendations for users.

## Key Features

- MovieLens dataset preprocessing and exploratory data analysis.
- User and movie feature engineering using PySpark.
- Two-Tower deep learning retrieval model.
- FAISS-based similarity search for candidate retrieval.
- Wide & Deep neural network for recommendation ranking.
- ONNX model export and NVIDIA Triton inference integration.
- FastAPI-based recommendation API.
- Redis caching integration.
- Streamlit dashboard.
- Airflow-based scheduled model retraining.
- Offline A/B testing and online A/B testing prototype.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | PyTorch |
| Data Processing | Pandas, PySpark |
| Retrieval | FAISS |
| Ranking | Wide & Deep |
| Model Deployment | ONNX, NVIDIA Triton |
| API | FastAPI |
| Dashboard | Streamlit |
| Caching | Redis |
| Workflow Orchestration | Apache Airflow |
| Containerization | Docker |
| Version Control | Git, GitHub |

## Project Architecture

1. Data collection and preprocessing.
2. Feature engineering and data preparation.
3. Two-Tower model generates candidate recommendations.
4. FAISS retrieves similar movie candidates.
5. Wide & Deep model ranks candidate movies.
6. Triton serves ONNX model inference.
7. FastAPI provides recommendation endpoints.
8. Redis supports caching.
9. Streamlit provides a dashboard.
10. Airflow schedules model retraining.

## Project Structure

```text
AIML03_Recommendation_Engine/
│
├── airflow/
│   └── dags/
├── models/
│   ├── faiss/
│   └── retrieval/
├── spark/
├── src/
│   ├── ab_testing.py
│   ├── clean_data.py
│   ├── cold_start.py
│   ├── evaluate_model.py
│   ├── faiss_retrieval.py
│   ├── ranking_model.py
│   ├── recommendation_api.py
│   ├── redis_cache.py
│   ├── split_data.py
│   ├── two_stage_recommendation.py
│   └── two_tower_model.py
│
├── triton_model_repository/
├── app.py
├── Dockerfile.airflow
├── docker-compose-airflow.yaml
├── export_onnx.py
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/bayyabhavana/AIML03-Recommendation-Engine.git
cd AIML03-Recommendation-Engine
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the required packages used by the project:

```bash
pip install numpy pandas scikit-learn torch fastapi uvicorn streamlit faiss-cpu onnx
```

Additional dependencies may be required for Spark, Redis, Airflow, and Triton.

### 4. Run the FastAPI application

```bash
uvicorn src.recommendation_api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Dashboard:

```text
http://localhost:8501
```

## Model Evaluation

The project includes offline evaluation using:

- Precision@10
- Recall@10
- NDCG@10
- RMSE
- MAE

The Wide & Deep model was evaluated during retraining with the following recorded results:

| Metric | Result |
|---|---:|
| RMSE | 1.1210 |
| MAE | 0.8649 |

These results are from a recorded training run and may vary across runs.

## A/B Testing

The project includes:

- Offline comparison of Two-Tower and Wide & Deep recommendation approaches.
- An online A/B testing prototype.
- User-group assignment.
- Impression and click event tracking.
- Click-through rate (CTR) reporting.

Note: The online A/B implementation was functionally tested using simulated events. It does not represent statistically validated real-user engagement.

## Deployment

The application has been tested locally using Docker, FastAPI, Streamlit, Redis, and NVIDIA Triton.

**Public deployment:** Pending.

## Limitations and Future Improvements

- Evaluate training on the complete MovieLens dataset.
- Improve cold-start recommendations.
- Optimize retrieval and ranking performance.
- Conduct real-user online A/B testing.
- Deploy the application to a public cloud platform.
- Add monitoring and production-scale model serving.

## Author

**Bayya Bhavana Goud**

GitHub: https://github.com/bayyabhavana

LinkedIn: https://www.linkedin.com/in/bayya-bhavanagoud/

## License

This project is developed for educational and portfolio purposes.