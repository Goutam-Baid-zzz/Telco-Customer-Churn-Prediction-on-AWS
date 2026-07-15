import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Config
BUCKET_NAME = "telcom-churn"
LOCAL_FILE_PATH = "data/Telco-Customer-Churn.csv"
S3_KEY = "raw/Telco-Customer-Churn.csv"
REGION = "ap-south-1"


def upload_csv_to_s3(local_path, bucket, s3_key, region):
    s3 = boto3.client("s3", region_name=region)

    try:
        s3.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded '{local_path}' to 's3://{bucket}/{s3_key}'")
    except FileNotFoundError:
        print(f"Local file not found: {local_path}")
    except NoCredentialsError:
        print("AWS credentials not found. Check your ~/.aws/credentials file.")
    except ClientError as e:
        print(f"Upload failed: {e}")


def verify_upload(bucket, prefix, region):
    s3 = boto3.client("s3", region_name=region)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" in response:
        print("\nFiles found in bucket under prefix:")
        for obj in response["Contents"]:
            size_kb = obj["Size"] / 1024
            print(f"  - {obj['Key']} ({size_kb:.1f} KB)")
    else:
        print("No files found — upload may have failed.")


if __name__ == "__main__":
    upload_csv_to_s3(LOCAL_FILE_PATH, BUCKET_NAME, S3_KEY, REGION)
    verify_upload(BUCKET_NAME, "raw/", REGION)