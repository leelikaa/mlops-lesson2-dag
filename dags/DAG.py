from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json
from sklearn.metrics import accuracy_score
import joblib
import sys
sys.path.append('/app/dags')
from data import load_data, prepare_data
from train import train
from test import test


default_args={
        'owner':'airflow',
        'depends_on_past': False,
        'start_date': datetime(2024, 3, 20),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),  # timedelta из пакета datetime
    }

file_path_to_model = '/app/model.pkl'
file_path_to_dataset = '/app/dataset/iris.csv'
file_path_to_data_train = '/app/dataset/iris_train.csv'
file_path_to_data_test = '/app/dataset/iris_test.csv'

dag = DAG(
    'hw_MLops2_e-kozlova-1',
    default_args = default_args,
    description='Logreg iris data',
    schedule_interval=timedelta(days=1),
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable= load_data,
    dag=dag,)

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable= prepare_data,
    op_kwargs={'csv_path': file_path_to_dataset},
    dag=dag,)

train_task = PythonOperator(
    task_id='train',
    python_callable= train,
    op_kwargs={'train_csv': file_path_to_data_train},
    dag=dag,)

test_task = PythonOperator(
    task_id='test',
    python_callable= test,
    op_kwargs={'model_path': file_path_to_model,'test_csv': file_path_to_data_test},
    dag=dag,)

load_data_task >> prepare_data_task >> train_task >> test_task

