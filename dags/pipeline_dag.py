from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'etl'))

from etl.load_data import load_data, load_config
from etl.preprocess_data import preprocess_data
from etl.train_model import train_model
from etl.evaluate_model import evaluate_model
from etl.save_results import save_results

# Функция callback при ошибке задачи
def failure_callback_function(context):
    task_instance = context.get('task_instance')
    print(f"Task {task_instance.task_id} failed on {task_instance.execution_date}")

# Загружаем конфигурацию один раз
config = load_config()

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=5),
    'on_failure_callback': failure_callback_function,
    'start_date': datetime(2023, 1, 1),
    'depends_on_past': False
}

with DAG(
    dag_id='ml_pipeline_breast_cancer',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'etl', 'airflow'],
    description='ML pipeline for breast cancer detection',
) as dag:

    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        op_kwargs={'config': config},
    )

    t2 = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        op_kwargs={'config': config},
    )

    t3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={'config': config},
    )

    t4 = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        op_kwargs={'config': config},
    )

    t5 = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
        op_kwargs={'config': config},
    )

    # Задаем зависимости
    t1 >> t2 >> t3 >> t4 >> t5
