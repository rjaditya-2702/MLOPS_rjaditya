# airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
from src.main import (
    create_and_save_dataset,
    preprocess_data,
    train_neural_network,
    test_model
)

# Define default arguments for your DAG
default_args = {
    'owner': 'Aditya Ratan',
    'start_date': datetime(2025, 1, 15),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_test_file_exists():
    """
    Check if test.txt file exists and branch accordingly
    Returns the task_id to execute next
    """
    # Get the directory where this DAG file is located
    dag_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(dag_dir, 'test', 'test.txt')

    if os.path.exists(test_file_path):
        print(f"Test file found at {test_file_path}")
        return 'test_model'  # Proceed to test_model task
    else:
        print(f"Test file not found at {test_file_path}")
        print("Terminating pipeline - no test file to process")
        return 'skip_testing'  # Go to skip_testing task

# Create a DAG instance
with DAG(
    'pytorch_circle_classifier',
    default_args=default_args,
    description='PyTorch Neural Network for Circle Classification',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Task 1: Create and save dataset
    data_loading_task = PythonOperator(
        task_id='create_dataset',
        python_callable=create_and_save_dataset,
        op_kwargs={'output_path': './data/circle_data.csv'},
    )

    # Task 2: Preprocess data (create train/test sets)
    data_preprocessing_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        op_kwargs={'input_path': './data/circle_data.csv'},
    )

    # Task 3: Train neural network model
    model_training_task = PythonOperator(
        task_id='train_model',
        python_callable=train_neural_network,
        op_args=[data_preprocessing_task.output],
        op_kwargs={'model_path': './models/circle_classifier.pth'},
    )

    # Branch Task: Check if test file exists
    check_test_file_task = BranchPythonOperator(
        task_id='check_test_file',
        python_callable=check_test_file_exists,
        trigger_rule='all_success',
    )

    # Task 4: Test the model with test.txt input (only runs if file exists)
    dag_dir = os.path.dirname(os.path.abspath(__file__))
    testing_task = PythonOperator(
        task_id='test_model',
        python_callable=test_model,
        op_kwargs={
            'model_path': './models/circle_classifier.pth',
            'test_file_path': os.path.join(dag_dir, 'test', 'test.txt')
        },
        trigger_rule='all_success',
    )

    # Dummy task for when test file doesn't exist
    skip_testing_task = DummyOperator(
        task_id='skip_testing',
        trigger_rule='all_success',
    )

    # Optional: End task to converge branches
    end_task = DummyOperator(
        task_id='end',
        trigger_rule='none_failed_or_skipped',
    )

    # Set task dependencies
    data_loading_task >> data_preprocessing_task >> model_training_task >> check_test_file_task
    
    # Branching logic
    check_test_file_task >> [testing_task, skip_testing_task]
    
    # Both branches converge to end
    testing_task >> end_task
    skip_testing_task >> end_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()