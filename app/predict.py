import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def generate_sample_data(self, n_samples=1000):
        """Generate sample e-commerce data"""
        np.random.seed(42)
        
        data = {
            'user_id': range(n_samples),
            'age': np.random.randint(18, 65, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'category_preference': np.random.choice(['Electronics', 'Fashion', 'Home', 'Sports'], n_samples),
            'price_range': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'purchase_frequency': np.random.randint(1, 20, n_samples)
        }
        
        # Create target (simplified recommendation categories)
        target = np.random.randint(0, 4, n_samples)  # 4 product categories
        
        return pd.DataFrame(data), target
    
    def preprocess_data(self, df):
        """Preprocess data for training"""
        # One-hot encoding for categorical variables
        df_encoded = pd.get_dummies(df, columns=['gender', 'category_preference', 'price_range'])
        return df_encoded
    
    def train_model(self):
        """Train the recommendation model"""
        # Generate sample data
        X, y = self.generate_sample_data()
        
        # Preprocess
        X_processed = self.preprocess_data(X.drop('user_id', axis=1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Start MLflow run
        with mlflow.start_run():
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Log parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.2)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "recommendation_model")
            
            print(f"Model trained successfully!")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
        
        return self.model
    
    def save_model(self, filepath='app/model.pkl'):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    model = trainer.train_model()
    trainer.save_model()
