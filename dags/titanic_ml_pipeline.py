from airflow import DAG
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import io
import os

def test_gcp_connection(**context):
    '''Test GCP connection'''
    try:
        hook = GCSHook(gcp_conn_id='google')
        client = hook.get_conn()
        bucket = client.bucket('my-bucket-2945')
        
        print(f' GCP connection successful')
        print(f' Bucket accessible: {bucket.exists()}')
        return True
    except Exception as e:
        print(f' Connection test failed: {e}')
        raise

def load_titanic_data(**context):
    '''Load Titanic data from PostgreSQL'''
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Load data
    df = postgres_hook.get_pandas_df('SELECT * FROM titanic;')
    print(f' Loaded {len(df)} rows from PostgreSQL')
    print(f' Columns: {df.columns.tolist()}')
    print(f' Sample data:')
    print(df.head())
    
    return df.to_json()

def preprocess_data(**context):
    '''Preprocess the Titanic data for ML'''
    # Get data from previous task
    data_json = context['task_instance'].xcom_pull(task_ids='load_titanic_data')
    df = pd.read_json(data_json)
    
    print(f' Original data shape: {df.shape}')
    print(f' Missing values:')
    print(df.isnull().sum())
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    
    le_embarked = LabelEncoder()
    df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])
    
    # Select features for training
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
    X = df[features]
    y = df['Survived']
    
    print(f' Features: {features}')
    print(f' Feature data shape: {X.shape}')
    print(f' Target distribution:')
    print(y.value_counts())
    
    return {
        'X': X.to_json(),
        'y': y.to_json(),
        'features': features
    }

def save_train_test_data(**context):
    '''Save train/test data to CSV files'''
    # Get preprocessed data
    data = context['task_instance'].xcom_pull(task_ids='preprocess_data')
    X = pd.read_json(data['X'])
    y = pd.read_json(data['y'], typ='series')
    
    # Combine features and target
    df_processed = X.copy()
    df_processed['Survived'] = y
    
    # Split data
    train_df, test_df = train_test_split(
        df_processed, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create artifacts directory
    artifacts_dir = '/usr/local/airflow/artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save train and test data
    train_df.to_csv(f'{artifacts_dir}/titanic_train.csv', index=False)
    test_df.to_csv(f'{artifacts_dir}/titanic_test.csv', index=False)
    
    print(f' Train data saved: {len(train_df)} rows')
    print(f' Test data saved: {len(test_df)} rows')
    print(f' Files saved to {artifacts_dir}/')
    
    return {
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'artifacts_dir': artifacts_dir
    }

def train_model(**context):
    '''Train Random Forest model'''
    # Get preprocessed data
    data = context['task_instance'].xcom_pull(task_ids='preprocess_data')
    X = pd.read_json(data['X'])
    y = pd.read_json(data['y'], typ='series')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f' Training set: {X_train.shape}')
    print(f' Test set: {X_test.shape}')
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    print(' Training Random Forest model...')
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f' Model trained successfully!')
    print(f' Training accuracy: {model.score(X_train, y_train):.4f}')
    print(f' Test accuracy: {accuracy:.4f}')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': data['features'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(' Feature Importance:')
    print(feature_importance)
    
    # Save model to artifacts
    artifacts_dir = '/usr/local/airflow/artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with open(f'{artifacts_dir}/titanic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f' Model saved to {artifacts_dir}/titanic_model.pkl')
    
    return {
        'model': pickle.dumps(model).hex(),
        'X_test': X_test.to_json(),
        'y_test': y_test.to_json(),
        'y_pred': pd.Series(y_pred).to_json(),
        'accuracy': accuracy,
        'feature_importance': feature_importance.to_json()
    }

def evaluate_model(**context):
    '''Evaluate the trained model'''
    # Get model results
    results = context['task_instance'].xcom_pull(task_ids='train_model')
    
    y_test = pd.read_json(results['y_test'], typ='series')
    y_pred = pd.read_json(results['y_pred'], typ='series')
    accuracy = results['accuracy']
    
    print(' MODEL EVALUATION RESULTS')
    print('=' * 50)
    print(f' Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    # Classification report
    print('\n Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('\n Confusion Matrix:')
    print('Predicted:  Died  Survived')
    print(f'Died:       {cm[0,0]:4d}  {cm[0,1]:8d}')
    print(f'Survived:   {cm[1,0]:4d}  {cm[1,1]:8d}')
    
    # Feature importance
    feature_importance = pd.read_json(results['feature_importance'])
    print('\n Top Features:')
    for _, row in feature_importance.head().iterrows():
        print(f'  {row["feature"]}: {row["importance"]:.4f}')
    
    # Save results to PostgreSQL
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Create results table
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS ml_results (
        run_date TIMESTAMP,
        model_type VARCHAR(50),
        accuracy FLOAT,
        test_samples INTEGER,
        feature_count INTEGER
    );
    '''
    
    postgres_hook.run(create_table_sql)
    
    # Insert results
    insert_sql = '''
    INSERT INTO ml_results (run_date, model_type, accuracy, test_samples, feature_count)
    VALUES (NOW(), 'RandomForest', %s, %s, %s);
    '''
    
    postgres_hook.run(insert_sql, parameters=[accuracy, len(y_test), len(results['feature_importance'])])
    
    print(f'\n Results saved to database!')
    print(f' ML Pipeline completed successfully!')
    
    return {
        'accuracy': accuracy,
        'test_samples': len(y_test),
        'model_type': 'RandomForest'
    }

# DAG definition
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'titanic_ml_pipeline',
    default_args=default_args,
    description='Titanic ML Pipeline - Train and Evaluate Model',
    schedule=None,
    catchup=False,
    tags=['titanic', 'ml', 'training']
) as dag:
    
    test_connection = PythonOperator(
        task_id='test_gcp_connection',
        python_callable=test_gcp_connection
    )
    
    load_data = PythonOperator(
        task_id='load_titanic_data',
        python_callable=load_titanic_data
    )
    
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )
    
    save_data = PythonOperator(
        task_id='save_train_test_data',
        python_callable=save_train_test_data
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
    
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )
    
    test_connection >> load_data >> preprocess >> save_data >> train >> evaluate
