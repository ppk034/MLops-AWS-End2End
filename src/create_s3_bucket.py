import boto3

# Initialize the S3 client
s3 = boto3.client('s3')

# Define the bucket name (must be unique across AWS)
bucket_name = 'mlproject-end2end-bucket'  # Use a unique name

# Create the bucket (no LocationConstraint needed for us-east-1)
try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' created successfully.")
except Exception as e:
    print(f"Error creating bucket: {e}")
