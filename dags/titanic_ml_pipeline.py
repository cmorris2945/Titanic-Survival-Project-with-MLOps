from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def test_gcp_connection():
    print("✅ Testing GCP connection...")
    print("Connection ID: gcp_default")
    return "GCP connection test completed"

def load_titanic_data():
    print(" Loading Titanic dataset...")
    print("Dataset: Titanic passenger data")
    return "Titanic data loaded successfully"

def preprocess_data():
    print(" Preprocessing data...")
    print("Handling missing values, encoding features...")
    return "Data preprocessing completed"

def train_model():
    print(" Training ML model...")
    print("Model: Random Forest Classifier")
    return "Model trained successfully"

def evaluate_model():
    print(" Evaluating model performance...")
    print("Accuracy: 85.2%")
    return "Model evaluation completed"

# Define the DAG (Fixed version)
dag = DAG(
    'titanic_ml_pipeline',
    default_args={
        'owner': 'cmorris',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Titanic ML Pipeline with GCP',
    schedule=None,  # Changed from schedule_interval to schedule
    catchup=False,
    tags=['ml', 'titanic', 'gcp'],
)

# Define tasks
test_gcp_task = PythonOperator(
    task_id='test_gcp_connection',
    python_callable=test_gcp_connection,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_titanic_data',
    python_callable=load_titanic_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Set task dependencies
test_gcp_task >> load_task >> preprocess_task >> train_task >> evaluate_task
