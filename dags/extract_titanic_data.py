from airflow import DAG
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import io

def validate_bucket_access(**context):
    '''Validate GCS bucket access and check for Titanic dataset'''
    try:
        hook = GCSHook(gcp_conn_id='google')
        client = hook.get_conn()
        
        # Check if bucket exists
        bucket = client.bucket('my-bucket-2945')
        bucket_exists = bucket.exists()
        print(f' Bucket "my-bucket-2945" exists: {bucket_exists}')
        
        # Check if file exists
        blob = bucket.blob('Titanic-Dataset.csv')
        file_exists = blob.exists()
        print(f' File "Titanic-Dataset.csv" exists: {file_exists}')
        
        if file_exists:
            print(f' File size: {blob.size} bytes')
            
        return {'bucket_exists': bucket_exists, 'file_exists': file_exists}
        
    except Exception as e:
        print(f' Bucket validation failed: {e}')
        raise

def list_files(**context):
    '''List all files in the GCS bucket'''
    hook = GCSHook(gcp_conn_id='google')
    client = hook.get_conn()
    bucket = client.bucket('my-bucket-2945')
    
    files = []
    for blob in bucket.list_blobs():
        files.append({
            'name': blob.name,
            'size': blob.size,
            'updated': str(blob.updated)
        })
        print(f' Found: {blob.name} ({blob.size} bytes)')
    
    return files

def process_file_list(**context):
    '''Process the list of files and identify Titanic dataset'''
    files = context['task_instance'].xcom_pull(task_ids='list_files')
    
    titanic_file = None
    for file in files:
        if 'titanic' in file['name'].lower() or 'Titanic-Dataset' in file['name']:
            titanic_file = file['name']
            print(f' Found Titanic dataset: {titanic_file}')
            break
    
    if not titanic_file:
        raise ValueError('Titanic dataset not found in bucket')
    
    return titanic_file

def download_file(**context):
    '''Download the Titanic dataset from GCS'''
    file_name = context['task_instance'].xcom_pull(task_ids='process_file_list')
    
    hook = GCSHook(gcp_conn_id='google')
    client = hook.get_conn()
    bucket = client.bucket('my-bucket-2945')
    blob = bucket.blob(file_name)
    
    # Download as string
    content = blob.download_as_text()
    print(f' Downloaded {file_name} ({len(content)} characters)')
    
    return content

def load_to_sql(**context):
    '''Load the Titanic data to PostgreSQL'''
    csv_content = context['task_instance'].xcom_pull(task_ids='download_file')
    
    # Parse CSV
    df = pd.read_csv(io.StringIO(csv_content))
    print(f' Dataset shape: {df.shape}')
    print(f' Columns: {df.columns.tolist()}')
    
    # Connect to PostgreSQL
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Create table
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS titanic (
        PassengerId INTEGER,
        Survived INTEGER,
        Pclass INTEGER,
        Name VARCHAR(255),
        Sex VARCHAR(10),
        Age FLOAT,
        SibSp INTEGER,
        Parch INTEGER,
        Ticket VARCHAR(50),
        Fare FLOAT,
        Cabin VARCHAR(50),
        Embarked VARCHAR(1)
    );
    '''
    
    postgres_hook.run(create_table_sql)
    postgres_hook.run('TRUNCATE TABLE titanic;')
    
    # Insert data
    from sqlalchemy import create_engine
    conn = postgres_hook.get_connection('postgres_default')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')
    
    df.to_sql('titanic', engine, if_exists='replace', index=False)
    print(f' Successfully loaded {len(df)} rows to titanic table')
    
    return len(df)

# DAG definition
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'extract_titanic_data',
    default_args=default_args,
    description='Extract Titanic data from GCS and load to PostgreSQL',
    schedule=None,
    catchup=False,
    tags=['titanic', 'gcs', 'etl']
) as dag:
    
    validate_task = PythonOperator(
        task_id='validate_bucket_access',
        python_callable=validate_bucket_access
    )
    
    list_task = PythonOperator(
        task_id='list_files',
        python_callable=list_files
    )
    
    process_task = PythonOperator(
        task_id='process_file_list',
        python_callable=process_file_list
    )
    
    download_task = PythonOperator(
        task_id='download_file',
        python_callable=download_file
    )
    
    load_task = PythonOperator(
        task_id='load_to_sql',
        python_callable=load_to_sql
    )
    
    validate_task >> list_task >> process_task >> download_task >> load_task
