from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
import sys
sys.path.insert(0, '/root/airflow')
from utils.utils import preprocessing, optimize_model,detect_new_data,read_new_data,branch_use_current_params,predictions
from datetime import datetime

args = {
    'owner': 'esteban126',
    'retries': 1,
    'start_date': datetime(2024, 1, 1),  # example fixed date
}

with DAG(
    dag_id='SodaI_Drinks', ## Name of DAG run
    default_args=args,
    description='MLops pipeline',
    schedule = None) as dag:

    # Task 1 - Just a simple print statement
    dummy_task = EmptyOperator(task_id='Start', retries=2)  

    # Branching task
    branch_task = BranchPythonOperator(
    task_id='new_or_old_data',
    python_callable=detect_new_data,
    provide_context=True,
    dag=dag
    )

    # Task 2
    branch_a = PythonOperator(
    task_id='read_new_data',
    python_callable=read_new_data
    )
    # Task 3
    branch_b = DummyOperator(task_id='dont_predict', dag=dag)

    task_preprocessing = PythonOperator(
    task_id='prep_data',
    python_callable=preprocessing
    )

    # Task 4
    task_branch_user_current_params = BranchPythonOperator(
    task_id='branch_use_current_params',
    python_callable=branch_use_current_params,
    provide_context=True,
    dag=dag
    )

    # Task 7
    task_optimize_model = PythonOperator(
    task_id='optimize_model',
    python_callable=optimize_model
    )    
    # Task 8
    task_predictions = PythonOperator(
    task_id='predictions',
    python_callable=predictions
    )    
    
    # Task 8
    final_dummy_task = EmptyOperator(task_id='End', retries=1)  
    # Define the workflow process
    

    dummy_task >> branch_task
    branch_task >> [branch_a,branch_b]
    branch_a >> task_preprocessing >> task_branch_user_current_params
    task_branch_user_current_params >> [task_optimize_model,task_predictions]
    task_optimize_model >> task_predictions
    task_predictions >> final_dummy_task
