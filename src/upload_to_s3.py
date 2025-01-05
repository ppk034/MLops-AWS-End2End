import boto3
import os

def upload_file_to_s3(file_path, bucket_name, object_name=None):
    """
    Uploads a file to an S3 bucket.

    :param file_path: Path to the file to upload
    :param bucket_name: S3 bucket name
    :param object_name: S3 object name (optional, defaults to file name)
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded {file_path} to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    # Define your bucket name and file to upload
    bucket_name = "mlproject-end2end-bucket"
    file_path = "../data/project_data.csv"  # Adjust this path if needed

    if os.path.exists(file_path):
        upload_file_to_s3(file_path, bucket_name)
    else:
        print(f"File {file_path} does not exist.")
