# Large-Scale Personalized Recommendation Engine

A two-stage movie recommendation system that generates personalized movie recommendations using machine learning, vector similarity search, and deep learning-based ranking.

## Live Demo

**Try the deployed application here:**

[Movie Recommendation Engine – Streamlit App](https://bayyabhavana-aiml03-recommendation-engine-app-lmzbu2.streamlit.app)

**GitHub Repository:**  
https://github.com/bayyabhavana/AIML03-Recommendation-Engine

---

## Project Overview

The Large-Scale Personalized Recommendation Engine is designed to recommend movies based on user preferences and movie interactions.

The system uses a two-stage recommendation architecture:

1. **Candidate Retrieval:** A Two-Tower deep learning model generates candidate movie recommendations.
2. **Ranking:** A Wide & Deep model ranks the retrieved candidates based on predicted user preferences.

FAISS is used for efficient similarity search, while Redis is included for caching support.

## Key Features

- Personalized movie recommendation architecture.
- Two-Tower retrieval model.
- Wide & Deep ranking model.
- FAISS-based vector similarity search.
- Movie popularity recommendations.
- Genre-based movie exploration.
- Model evaluation dashboard.
- FastAPI recommendation endpoints.
- Redis caching integration.
- Online A/B testing prototype.
- ONNX model export.
- NVIDIA Triton inference integration.
- Apache Airflow scheduled model retraining.
- Streamlit web application.

## System Architecture

The project follows a two-stage recommendation pipeline:

**User and Movie Data → Data Preprocessing → Feature Engineering → Two-Tower Retrieval → FAISS Candidate Search → Wide & Deep Ranking → Personalized Recommendations**

Supporting components:

- **Apache Spark:** Data processing and feature engineering.
- **FAISS:** Vector similarity search.
- **Redis:** Caching support.
- **FastAPI:** Recommendation API.
- **ONNX and NVIDIA Triton:** Model inference integration.
- **Apache Airflow:** Scheduled model retraining.
- **Streamlit:** Interactive application and dashboard.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy, Apache Spark |
| Machine Learning | PyTorch |
| Recommendation Models | Two-Tower, Wide & Deep |
| Vector Search | FAISS |
| Model Format | ONNX |
| Model Serving | NVIDIA Triton |
| API Development | FastAPI |
| Caching | Redis |
| Workflow Orchestration | Apache Airflow |
| Containerization | Docker |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

## Project Structure

```text
AIML03-Recommendation-Engine/
│
├── airflow/
│   └── dags/
│       └── recommendation_retraining.py
│
├── data/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── evaluation_results.csv
│   ├── ab_testing_results.csv
│   └── cold_start_popular.csv
│
├── spark/
│
├── src/
│   ├── ranking_model.py
│   ├── recommendation_api.py
│   ├── two_stage_recommendation.py
│   └── ab_testing.py
│
├── triton_model_repository/
│
├── app.py
├── requirements.txt
├── Dockerfile.airflow
├── docker-compose-airflow.yaml
├── export_onnx.py
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bayyabhavana/AIML03-Recommendation-Engine.git
```

### 2. Navigate to the Project Directory

```bash
cd AIML03-Recommendation-Engine
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows:**

```powershell
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open locally at:

```text
http://localhost:8501
```

## Application Modules

### 1. Home
Displays the recommendation engine overview, dataset information, recommendation model, and caching information.

### 2. Popular Movies
Displays the top popular movies with their genres, average ratings, and rating counts.

### 3. Genre Recommendations
Allows users to select a movie genre and explore matching movies.

### 4. Model Evaluation
Displays the available model evaluation metrics and results.

## API Endpoints

The FastAPI service includes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `/docs` | Interactive API documentation |
| `/triton/health` | Triton health check |
| `/triton/predict` | Triton model prediction |
| `/ab/recommend/{user_id}` | A/B assigned recommendations |
| `/ab/click` | Record recommendation clicks |
| `/ab/report` | View A/B testing report |

These endpoints are available when the corresponding local API services are running.

## Model Evaluation

The project evaluates recommendation performance using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Precision@10
- Recall@10
- NDCG@10

The recorded Wide & Deep retraining run achieved:

| Metric | Result |
|---|---:|
| RMSE | 1.1210 |
| MAE | 0.8649 |

An offline A/B comparison was also conducted between the Two-Tower and Wide & Deep approaches.

## Deployment

The Streamlit dashboard is deployed using **Streamlit Community Cloud**.

**Live Application:**  
https://bayyabhavana-aiml03-recommendation-engine-app-lmzbu2.streamlit.app

The dashboard is publicly accessible. The complete supporting infrastructure, including Airflow, Redis, and Triton, was developed and tested locally.

## Important Notes and Limitations

- The implementation uses PyTorch for the recommendation models rather than TensorFlow Recommenders.
- The recorded ranking-model experiment used a training sample of 80,000 records; full MovieLens 25M training has not been demonstrated.
- The online A/B testing prototype was functionally tested using simulated click events. Its CTR is not a result from a real user experiment.
- Airflow was configured for local demonstration using SQLite and SequentialExecutor.
- The public Streamlit deployment provides the dashboard functionality; it does not mean all local infrastructure services are publicly hosted.

## Future Enhancements

- Scale model training to the complete MovieLens 25M dataset.
- Improve recommendation accuracy and ranking performance.
- Implement robust cold-start recommendations.
- Add real-time user interaction tracking.
- Conduct statistically valid online A/B experiments.
- Deploy the complete recommendation infrastructure to cloud services.
- Improve model monitoring and automated retraining.

## Conclusion

This project demonstrates a personalized movie recommendation pipeline combining deep learning, vector similarity search, data processing, model serving, API development, workflow orchestration, and interactive dashboard deployment.

It provides practical experience in building and integrating machine learning components into a recommendation system.

---

**Project:** Large-Scale Personalized Recommendation Engine  
**Category:** Artificial Intelligence and Machine Learning  
**Application:** Movie Recommendation System