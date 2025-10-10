import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys  
import redis
import json

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, train_data_path, test_data_path, feature_store: RedisFeatureStore):
       self.train_data_path = train_data_path
       self.test_data_path = test_data_path
       self.data = None
       self.test_data = None
       self.X_train = None
       self.X_test = None
       self.y_train = None
       self.y_test = None
       self.X_resampled = None
       self.y_resampled = None
       self.feature_store = feature_store
       logger.info("Your Data Processing is initiated....")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Read the data successfully")
            logger.info(f"Columns in data: {self.data.columns.tolist()}")  # Debug line
        except Exception as e:
            logger.error(f"Error while reading data {e}")
            raise CustomException(str(e), sys)  # Fixed: added sys
        
    def preprocess_data(self):
        try:
            # Check what columns actually exist
            logger.info(f"Available columns: {self.data.columns.tolist()}")
            
            # Handle missing values for existing columns only
            if 'Age' in self.data.columns:
                self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
            
            # Skip Embarked if it doesn't exist (already processed)
            if 'Embarked' in self.data.columns:
                self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
                self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes
            
            if 'Fare' in self.data.columns:
                self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())
            
            # Handle Sex encoding if not already done
            if 'Sex' in self.data.columns and self.data['Sex'].dtype == 'object':
                self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
            
            # Create new features only if base columns exist
            if 'SibSp' in self.data.columns and 'Parch' in self.data.columns:
                self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1 
                self.data["Isalone"] = (self.data["FamilySize"] == 1).astype(int)
            
            if 'Cabin' in self.data.columns:
                self.data["HasCabin"] = self.data["Cabin"].notnull().astype(int)
            
            if 'Name' in self.data.columns:
                self.data["Title"] = self.data["Name"].str.extract(' ([A-Za-z]+)\.', expand=False).map(
                    {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(4)
            
            if 'Pclass' in self.data.columns and 'Fare' in self.data.columns:
                self.data["Pclass_fare"] = self.data["Pclass"] * self.data["Fare"]
            
            if 'Age' in self.data.columns and 'Fare' in self.data.columns:
                self.data["Age_fare"] = self.data["Age"] * self.data["Fare"]

            logger.info("Data Preprocessing done....")
            logger.info(f"Final columns: {self.data.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error while preprocessing you punk! here...{e}")
            raise CustomException(str(e), sys)  # Fixed: added sys
        
    def handle_imbalance_data(self):
        try:
            # Use only columns that exist
            available_features = []
            potential_features = ['Pclass', 'Sex_encoded', 'Age', 'Fare', 'Embarked_encoded', 
                                'FamilySize', 'Isalone', 'HasCabin', 'Title', 'Pclass_fare', 'Age_fare']
            
            for feature in potential_features:
                if feature in self.data.columns:
                    available_features.append(feature)
            
            # If Sex_encoded doesn't exist but Sex does, use Sex
            if 'Sex_encoded' not in available_features and 'Sex' in self.data.columns:
                available_features.append('Sex')
                available_features.remove('Sex_encoded') if 'Sex_encoded' in available_features else None
            
            # If Embarked_encoded doesn't exist but Embarked does, use Embarked  
            if 'Embarked_encoded' not in available_features and 'Embarked' in self.data.columns:
                available_features.append('Embarked')
                available_features.remove('Embarked_encoded') if 'Embarked_encoded' in available_features else None
            
            logger.info(f"Using features for SMOTE: {available_features}")
            
            X = self.data[available_features]
            y = self.data['Survived']

            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)

            logger.info("handled imbalanced data successfully... ")
        except Exception as e:
            logger.error(f"Error while balancing your stupid data.... {e}")
            raise CustomException(str(e), sys)  # Fixed: added sys
        
    def store_feature_in_redis(self):
        try:
            batch_data = {}
            
            # Check if PassengerId exists, if not use index
            if 'PassengerId' in self.data.columns:
                id_column = 'PassengerId'
            else:
                # Use index as entity_id if PassengerId doesn't exist
                logger.info("PassengerId not found, using index as entity_id")
                id_column = None
            
            for idx, row in self.data.iterrows():
                # Use PassengerId if available, otherwise use index
                entity_id = row[id_column] if id_column else idx
                
                # Only include features that exist
                features = {}
                
                # Always include the entity_id
                features["entity_id"] = entity_id
                
                # Check each feature before adding
                possible_features = {
                    "Age": "Age",
                    "Fare": "Fare", 
                    "Pclass": "Pclass",
                    "Sex": "Sex",
                    "Sex_encoded": "Sex_encoded",
                    "Embarked": "Embarked",
                    "Embarked_encoded": "Embarked_encoded",
                    "FamilySize": "FamilySize",
                    "Isalone": "Isalone",
                    "HasCabin": "HasCabin",
                    "Title": "Title",
                    "Pclass_fare": "Pclass_fare",
                    "Age_fare": "Age_fare",
                    "Survived": "Survived"
                }
                
                for feature_name, column_name in possible_features.items():
                    if column_name in self.data.columns:
                        features[feature_name] = row[column_name]
                
                batch_data[entity_id] = features
            
            self.feature_store.store_batch_features(batch_data)
            logger.info(f"Stored {len(batch_data)} records in feature store")
            logger.info("Data has been fed into feature store workflow...")
            
        except Exception as e:
            logger.error(f"Error while trying to run feature store idiot... {e}")
            raise CustomException(str(e), sys)
        
    def retrieve_feature_redis_store(self, entity_id):
        features = self.feature_store.get_features(entity_id)
        if features:
            return features
        return None
    
    def run(self):
        try:
            logger.info("Starting the data processing pipeline stupid....")
            self.load_data()
            self.preprocess_data()
            self.handle_imbalance_data()
            self.store_feature_in_redis()
            logger.info("End of pipeline data processing. (Big Whoop!)")
        except Exception as e:
            logger.error(f"Error while data processing pipeline fool! {e}")
            raise CustomException(str(e), sys)  # Fixed: added sys

if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, feature_store)
    data_processor.run()
    print(data_processor.retrieve_feature_redis_store(entity_id=50)) 