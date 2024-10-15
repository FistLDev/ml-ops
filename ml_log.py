import mlflow
import mlflow.sklearn
from mlflow.utils.file_utils import TempDir
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import boto3
import json
import os

EXPERIMENT_NAME = "Симоненков Алексей Витальевич"
PARENT_RUN_NAME = "maldey2"
BUCKET_NAME = "hw2_bucket"
S3_CONNECTION = ""

def log_model_and_metrics(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    with TempDir() as tmp:
        model_path = tmp.path(model_name)
        mlflow.sklearn.save_model(model, model_path)
        mlflow.log_artifact(model_path, artifact_path=model_name)
    
def load_data():
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    experiments = [exp.name for exp in mlflow.list_experiments()]
    if EXPERIMENT_NAME not in experiments:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
        with mlflow.start_run(run_name="LinearRegression", nested=True):
            model = LinearRegression()
            model.fit(X_train, y_train)
            log_model_and_metrics(model, X_test, y_test, "LinearRegression")
        
        with mlflow.start_run(run_name="DecisionTree", nested=True):
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            log_model_and_metrics(model, X_test, y_test, "DecisionTree")
        
        with mlflow.start_run(run_name="RandomForest", nested=True):
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            log_model_and_metrics(model, X_test, y_test, "RandomForest")

if __name__ == "__main__":
    main()