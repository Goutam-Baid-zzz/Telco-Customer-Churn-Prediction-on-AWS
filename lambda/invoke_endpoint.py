import json
import boto3

runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")
ENDPOINT_NAME = "telco-churn-endpoint"

def lambda_handler(event, context):
    try:
        # API Gateway (proxy integration) sends the body as a JSON string
        body = json.loads(event["body"]) if "body" in event else event

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(body)
        )

        result = json.loads(response["Body"].read().decode())

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }