import datetime as dt
import os, sys
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_PATH = "/opt/airflow/airflow_hw"
os.environ["PROJECT_PATH"] = PROJECT_PATH

def run_pipeline():
    if PROJECT_PATH not in sys.path:
        sys.path.insert(0, PROJECT_PATH)
    from modules.pipeline import pipeline
    pipeline()

def run_predict():
    if PROJECT_PATH not in sys.path:
        sys.path.insert(0, PROJECT_PATH)
    from modules.predict import predict
    predict()

default_args = {
    "owner": "airflow",
    "start_date": dt.datetime(2022, 6, 10),
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=1),
    "depends_on_past": False,
}

with DAG(
    dag_id="car_price_prediction",
    schedule="0 15 * * *",
    default_args=default_args,
    catchup=False,
    tags=["hw"],
) as dag:
    t_pipeline = PythonOperator(task_id="pipeline", python_callable=run_pipeline)
    t_predict  = PythonOperator(task_id="predict",  python_callable=run_predict)
    t_pipeline >> t_predict
