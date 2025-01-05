import sagemaker
from sagemaker import get_execution_role
from sagemaker.xgboost import XGBoostModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Get execution role
role = 'arn:aws:iam::651706748381:role/SageMakerExecutionRole' 

# Define S3 path for the model file
model_s3_path = "s3://mlproject-end2end-bucket/xgboost_model.tar.gz"

# Define the model using the XGBoost pre-built container
xgboost_model = XGBoostModel(
    model_data=model_s3_path,  # Path to the trained model in S3
    role='arn:aws:iam::651706748381:role/SageMakerExecutionRole',
    entry_point='inference.py',  # We'll create this script next
    framework_version='1.5-1',
    sagemaker_session=sagemaker_session
)

# Deploy the model to create a real-time endpoint
predictor = xgboost_model.deploy(instance_type='ml.m5.large', initial_instance_count=1)

print("Model deployed successfully.")
