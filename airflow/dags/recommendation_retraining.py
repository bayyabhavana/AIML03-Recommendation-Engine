
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"

with DAG(
    dag_id="recommendation_retraining",
    description="Scheduled retraining of the Wide and Deep recommendation model",
    start_date=datetime(2026, 11, 6),
    schedule="0 2 * * 1",
    catchup=False,
    tags=["recommendation", "machine-learning"],
) as dag:

    retrain_model = BashOperator(
        task_id="retrain_wide_deep_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/ranking_model.py"
        ),
    )

    retrain_model
