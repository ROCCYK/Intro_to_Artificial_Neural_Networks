import mlflow.deployments

experiment_id = '1'
run_id = 'f5f937e1d177403b97253cebe530db66'
region = 'us-east-1'
aws_id = 'give your AWS ID'
arn = 'give your ARN'

deployment_name = 'model-application-iris'
model_uri = 'runs:/f5f937e1d177403b97253cebe530db66/random-forest-model'# f'mlruns/{experiment_id}/{run_id}/artifacts/random-forest-model'

tag_id = '2.8.0'
deployment_client = mlflow.deployments.get_deploy_client("sagemaker:/" + region)

#image_ecr_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id

image_ecr_url = "give your ECR_URL"
deployment_client.create_deployment(
    name=deployment_name,
    model_uri=model_uri,
    config={"image_url": image_ecr_url,
            "execution_role_arn": arn
            }
            )