from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from airflow.hooks.S3_hook import S3Hook
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
import os

BUCKET = Variable.get("S3_BUCKET")

default_args = {
    'owner': 'Симоненков Алексей Витальевич',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

def create_dag(dag_id, model_class, model_name):
    def init(**kwargs):
        kwargs['ti'].xcom_push(key='metrics', value={'timestamp': datetime.now(), 'model_name': model_name})

    def get_data(**kwargs):
        metrics = kwargs['ti'].xcom_pull(key='metrics')
        metrics['data_start'] = datetime.now()

        data = fetch_california_housing()
        metrics['data_end'] = datetime.now()
        metrics['data_size'] = data.data.shape

        s3 = S3Hook('s3_connection')
        s3.load_string(json.dumps(data.data.tolist()), f'{model_name}/datasets/data.json', BUCKET, replace=True)
        s3.load_string(json.dumps(data.target.tolist()), f'{model_name}/datasets/target.json', BUCKET, replace=True)

        kwargs['ti'].xcom_push(key='metrics', value=metrics)

    def prepare_data(**kwargs):
        metrics = kwargs['ti'].xcom_pull(key='metrics')
        metrics['prepare_start'] = datetime.now()

        s3 = S3Hook('s3_connection')
        data = json.loads(s3.read_key(f'{model_name}/datasets/data.json', BUCKET))
        target = json.loads(s3.read_key(f'{model_name}/datasets/target.json', BUCKET))

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        metrics['prepare_end'] = datetime.now()
        metrics['features'] = "all features scaled"

        s3.load_string(json.dumps(data.tolist()), f'{model_name}/datasets/scaled_data.json', BUCKET, replace=True)
        s3.load_string(json.dumps(target), f'{model_name}/datasets/scaled_target.json', BUCKET, replace=True)

        kwargs['ti'].xcom_push(key='metrics', value=metrics)

    def train_model(**kwargs):
        metrics = kwargs['ti'].xcom_pull(key='metrics')
        metrics['train_start'] = datetime.now()

        s3 = S3Hook('s3_connection')
        data = json.loads(s3.read_key(f'{model_name}/datasets/scaled_data.json', BUCKET))
        target = json.loads(s3.read_key(f'{model_name}/datasets/scaled_target.json', BUCKET))

        model = model_class()
        model.fit(data, target)
        predictions = model.predict(data)
        metrics['train_end'] = datetime.now()
        metrics['mse'] = mean_squared_error(target, predictions)

        s3.load_string(json.dumps(model.get_params()), f'{model_name}/results/model_params.json', BUCKET, replace=True)

        kwargs['ti'].xcom_push(key='metrics', value=metrics)

    def save_results(**kwargs):
        metrics = kwargs['ti'].xcom_pull(key='metrics')
        s3 = S3Hook('s3_connection')
        s3.load_string(json.dumps(metrics), f'{model_name}/results/metrics.json', BUCKET, replace=True)

    dag = DAG(dag_id=dag_id, default_args=default_args, schedule_interval='0 1 * * *', start_date=datetime(2024, 10, 15), tags=['mlops'])

    with dag:
        init_task = PythonOperator(task_id='init', python_callable=init, provide_context=True)
        get_data_task = PythonOperator(task_id='get_data', python_callable=get_data, provide_context=True)
        prepare_data_task = PythonOperator(task_id='prepare_data', python_callable=prepare_data, provide_context=True)
        train_model_task = PythonOperator(task_id='train_model', python_callable=train_model, provide_context=True)
        save_results_task = PythonOperator(task_id='save_results', python_callable=save_results, provide_context=True)

        init_task >> get_data_task >> prepare_data_task >> train_model_task >> save_results_task

    return dag

DAG_1 = create_dag('linear_regression_dag', LinearRegression, 'LinearRegression')
DAG_2 = create_dag('decision_tree_dag', DecisionTreeRegressor, 'DecisionTree')
DAG_3 = create_dag('random_forest_dag', RandomForestRegressor, 'RandomForest')