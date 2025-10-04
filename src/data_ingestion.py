import psycopg2
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
from config.database_config import DB_CONFIG
from config.paths_config import *

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, db_params, output_dir):
        self.db_params = db_params
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = os.path.join(self.output_dir, 'raw')
        self.processed_dir = os.path.join(self.output_dir, 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def connect_to_db(self):
        try:
            conn = psycopg2.connect(
                host=self.db_params['host'],
                port=self.db_params['port'],
                dbname=self.db_params['dbname'],
                user=self.db_params['user'],
                password=self.db_params['password']
            )
            logger.info("Database connection is now established. So don't screw this up this time punk!")
            return conn
        except Exception as e:
            logger.error(f"You have an error genius! Check the logs....{e}")
            raise CustomException(str(e), sys)
        
    def extract_data(self):
        try:
            conn = self.connect_to_db()
            # Use correct column name: PassengerId
            query = "SELECT * FROM public.titanic ORDER BY \"PassengerId\""
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Data extracted from DB: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Hey dipshit! You got an error: {e}")
            raise CustomException(str(e), sys)
    
    def preprocess_data(self, df):
        """Preprocess data for ML training"""
        try:
            df_processed = df.copy()
            
            # Handle missing values using correct column names
            df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
            df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
            df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
            
            # Encode categorical variables
            le_sex = LabelEncoder()
            df_processed['Sex_encoded'] = le_sex.fit_transform(df_processed['Sex'])
            
            le_embarked = LabelEncoder()
            df_processed['Embarked_encoded'] = le_embarked.fit_transform(df_processed['Embarked'])
            
            # Select features for ML training
            ml_features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded', 'Survived']
            df_ml = df_processed[ml_features].copy()
            
            logger.info(f"ML features selected: {df_ml.columns.tolist()}")
            logger.info("Data preprocessing completed")
            return df_ml
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise CustomException(str(e), sys)
        
    def save_data(self, df_raw, df_processed):
        try:
            # Split raw data (with all original columns including Name, Ticket, Cabin)
            train_raw, test_raw = train_test_split(
                df_raw, 
                test_size=0.2, 
                random_state=42, 
                stratify=df_raw['Survived']
            )
            
            # Split processed data (ML-ready features only)
            train_processed, test_processed = train_test_split(
                df_processed, 
                test_size=0.2, 
                random_state=42, 
                stratify=df_processed['Survived']
            )
            
            # Save raw data (human-readable with names, tickets, etc.)
            raw_train_path = os.path.join(self.raw_dir, 'titanic_train_raw.csv')
            raw_test_path = os.path.join(self.raw_dir, 'titanic_test_raw.csv')
            train_raw.to_csv(raw_train_path, index=False)
            test_raw.to_csv(raw_test_path, index=False)
            
            # Save processed data (ML-ready features)
            processed_train_path = os.path.join(self.processed_dir, 'titanic_train.csv')
            processed_test_path = os.path.join(self.processed_dir, 'titanic_test.csv')
            train_processed.to_csv(processed_train_path, index=False)
            test_processed.to_csv(processed_test_path, index=False)
            
            # Also save to original paths for backward compatibility
            train_processed.to_csv(TRAIN_PATH, index=False)
            test_processed.to_csv(TEST_PATH, index=False)

            logger.info(f"Raw data saved - Train: {len(train_raw)} rows, Test: {len(test_raw)} rows")
            logger.info(f"Raw columns: {train_raw.columns.tolist()}")
            logger.info(f"Processed data saved - Train: {len(train_processed)} rows, Test: {len(test_processed)} rows")
            logger.info(f"Processed columns: {train_processed.columns.tolist()}")
            logger.info("Data splitting and saving done idiot...")
            
        except Exception as e:
            logger.error(f"There is a error fool! {e}")
            raise CustomException(str(e), sys)
    
    def run(self):
        try:
            logger.info("Data Ingestion Pipeline started...")
            
            # Extract raw data
            df_raw = self.extract_data()
            
            # Preprocess for ML
            df_processed = self.preprocess_data(df_raw)
            
            # Save both versions
            self.save_data(df_raw, df_processed)
            
            logger.info("End of data ingestion poopline.")
            
        except Exception as e:
            logger.error(f"Error while data ingestion pipeline {e}")
            raise CustomException(str(e), sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion.run()
