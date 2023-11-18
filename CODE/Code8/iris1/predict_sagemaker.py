import mlflow.deployments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import json
import boto3

region = 'us-east-1'
deployment_name = 'model-application-iris'
deployment_client = mlflow.deployments.get_deploy_client("sagemaker:/" + region)

deployment_info = deployment_client.get_deployment(name=deployment_name)
print(f"MLflow SageMaker Deployment status is: {deployment_info['EndpointStatus']}")

iris = datasets.load_iris()
x = iris.data
y = iris.target
query_df = pd.DataFrame(x).iloc[[55]]
actual = y[55]
prediction1 = deployment_client.predict(deployment_name, query_df)
print(f"Prediction response: {prediction1}")
print("Actual value:", actual)
