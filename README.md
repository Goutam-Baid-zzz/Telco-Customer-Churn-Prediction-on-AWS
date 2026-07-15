# 📡 Signal Check — Telco Customer Churn Prediction on AWS

An end-to-end MLOps pipeline that takes a raw customer dataset all the way to a live, publicly callable prediction API and a web frontend — built entirely on native AWS services.

<p align="left">
  <img src="https://img.shields.io/badge/AWS-S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-Glue-8C4FFF?style=for-the-badge&logo=amazonaws&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-SageMaker%20AI-01A88D?style=for-the-badge&logo=amazonaws&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-Lambda-FF9900?style=for-the-badge&logo=awslambda&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-API%20Gateway-FF4F8B?style=for-the-badge&logo=amazonapigateway&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-IAM-DD344C?style=for-the-badge&logo=amazoniam&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-1.5.2-0057B8?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/JavaScript-Vanilla-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" />
</p>

## Contents

- [Project goal](#-project-goal)
- [Architecture](#-architecture)
- [AWS services used](#️-aws-services-used)
- [Repository structure](#-repository-structure)
- [Setup guide](#-setup-guide-from-scratch)
- [Model performance](#-model-performance)
- [Screenshots](#️-screenshots)
- [Notes & caveats](#️-notes--caveats)
- [Possible extensions](#-possible-extensions)

---

## 🎯 Project goal

This is **not** a machine learning modeling exercise. The goal was never to squeeze out the highest possible accuracy, tune hyperparameters exhaustively, or benchmark multiple algorithms — a default, unoptimized XGBoost classifier was intentionally used as-is.

**The actual goal was to design, build, and deploy a complete, production-shaped ML pipeline on AWS** — ingestion, transformation, training, real-time hosting, and a consumer-facing interface — wired together with genuinely separate AWS-native services the way a real system would be, including working through the IAM, permissions, and infrastructure issues that come with that.

> This is an AWS deployment and pipeline-engineering project that happens to use a churn model, not a churn-modeling project that happens to use AWS.

---

## 🏗 Architecture

<p align="center">
  <img src="screenshots/architecture-diagram.svg" alt="Signal Check architecture diagram" width="480"/>
</p>

**Data flow, in words:**

| Step | What happens |
|---|---|
| 1 | Raw Telco Customer Churn CSV is pushed to S3 via a local `boto3` script |
| 2 | An AWS Glue ETL job cleans and transforms the data (type casting, target encoding, binary encoding), writing the result back to S3 |
| 3 | A SageMaker AI training job (run from a managed notebook instance) trains an XGBoost classifier on the processed data and stores the model artifact in S3 |
| 4 | The trained model is deployed to a real-time SageMaker AI endpoint |
| 5 | A Lambda function wraps `sagemaker-runtime.invoke_endpoint()` behind a simple JSON contract |
| 6 | API Gateway exposes that Lambda as a public HTTPS `POST /predict` route, with CORS enabled |
| 7 | A browser-based frontend collects a customer profile and calls that endpoint, displaying the churn prediction live |

---

## ☁️ AWS services used

| Service | Role |
|---|---|
| **S3** | Object storage for raw CSV, ETL output, and trained model artifacts — split into `raw/`, `processed/`, and `model-artifacts/` prefixes |
| **Glue** | Serverless Spark ETL job — type cleaning, target/binary encoding, writes cleaned data back to S3 |
| **SageMaker AI** | Managed notebook instance for training; hosts the real-time inference endpoint (XGBoost via `SKLearnModel` + custom `model_fn`/`predict_fn`) |
| **Lambda** | Thin invocation layer between API Gateway and the SageMaker endpoint; formats requests/responses and sets CORS headers |
| **API Gateway** | Public HTTP API exposing a `POST /predict` route backed by the Lambda function |
| **IAM** | Scoped roles per service — `TelcoChurn-Glue-Role`, `TelcoChurn-SageMaker-Role`, `TelcoChurn-Lambda-Role` — plus a programmatic IAM user for local `boto3` access |

---

## 📁 Repository structure

```
telco-churn-aws/
├── data/
│   └── Telco-Customer-Churn.csv        # source dataset (Kaggle, IBM sample)
├── ingestion/
│   └── upload_to_s3.py                 # boto3 script — local CSV → S3 raw/
├── glue_job/
│   └── churn_etl_job.py                # Glue Spark ETL script
├── sagemaker/
│   ├── train_launch.ipynb              # notebook: launches training + deploys endpoint
│   └── src/
│       ├── train.py                    # training entry point + model_fn/predict_fn for serving
│       └── requirements.txt            # xgboost pinned version (only extra dep needed)
├── lambda/
│   └── invoke_endpoint.py              # Lambda handler — API Gateway → SageMaker endpoint
├── frontend/
│   └── index.html                      # Signal Check UI — single-file HTML/CSS/JS
├── screenshots/                        # console screenshots + architecture diagram
├── infra_notes/
│   └── resource_names.md               # bucket/role/endpoint names for reference
└── README.md
```

---

## 🚀 Setup guide (from scratch)

**Prerequisites**
- An AWS account
- Python 3.9+ and `pip` installed locally
- The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) saved locally as `data/Telco-Customer-Churn.csv`

**1. AWS setup (IAM + S3)**
- Create an S3 bucket with three prefixes: `raw/`, `processed/`, `model-artifacts/`
- Create three IAM roles: `TelcoChurn-Glue-Role`, `TelcoChurn-SageMaker-Role`, `TelcoChurn-Lambda-Role`, each trusted by its respective service and scoped to your bucket
- Create an IAM user (or use existing credentials) with S3 access for local `boto3` calls, and configure `~/.aws/credentials` and `~/.aws/config`

**2. Upload the raw data**
```bash
pip install boto3
python ingestion/upload_to_s3.py
```
This pushes `Telco-Customer-Churn.csv` to `s3://<your-bucket>/raw/`.

**3. Run the Glue ETL job**
- Create a Glue job named `Telco-Churn-ETL`, paste in `glue_job/churn_etl_job.py`
- Point it at your bucket's `raw/` and `processed/` prefixes
- Run the job — cleaned data lands in `s3://<your-bucket>/processed/`

**4. Train the model**
- Launch a SageMaker AI notebook instance, attach `TelcoChurn-SageMaker-Role`
- Upload `sagemaker/train_launch.ipynb` and the `sagemaker/src/` folder
- Run the notebook — this trains an XGBoost classifier and uploads the model artifact to `s3://<your-bucket>/model-artifacts/`

**5. Deploy the endpoint**
- Still in the notebook, deploy the trained model to a real-time SageMaker AI endpoint (see the deploy cell in `train_launch.ipynb`)
- Confirm it reaches `InService` status

**6. Wire up Lambda + API Gateway**
- Create a Lambda function, paste in `lambda/invoke_endpoint.py`, attach `TelcoChurn-Lambda-Role`
- Create an HTTP API in API Gateway with a `POST /predict` route integrated to that Lambda
- Enable CORS (`Access-Control-Allow-Origin: *`, headers: `content-type`, methods: `POST, OPTIONS`)
- Copy the stage's invoke URL

**7. Run the frontend**
- Open `frontend/index.html`, replace the `API_URL` constant with your invoke URL + `/predict`
- Serve it locally (`python -m http.server` or VS Code Live Server — don't open via `file://`, it breaks CORS)
- Fill in a customer profile and click **Run diagnostic**

---

## 📊 Model performance

A default, untuned XGBoost classifier was used — no hyperparameter search, no ensembling, no feature engineering beyond basic encoding. This was deliberate, since model performance was not the point of the project.

| Metric | Score |
|---|---|
| Accuracy | 0.7906 |
| Precision | 0.6262 |
| Recall | 0.5241 |
| AUC | 0.8343 |

---

## 🖼️ Screenshots

| | |
|---|---|
| **IAM Roles** ![IAM roles](screenshots/00-iam-roles.png) | **S3 Bucket Structure** ![S3 structure](screenshots/01-s3-bucket-structure.png) |
| **Glue ETL Script** ![Glue script](screenshots/02-glue-etl-script.png) | **Glue Job — Succeeded** ![Glue run](screenshots/02-glue-job-run-success.png) |
| **SageMaker Notebook Instance** ![Notebook](screenshots/03-sagemaker-notebook-instance.png) | **Training Script in Jupyter** ![Train script](screenshots/03-jupyter-train-script.png) |
| **API Gateway Routes** ![Routes](screenshots/05-api-gateway-routes.png) | **Lambda Test — 200 OK** ![Lambda test](screenshots/05-lambda-test-success.png) |

---

## ⚠️ Notes & caveats

- The SageMaker endpoint (`ml.m5.large`) bills hourly while `InService` — delete it between demos to avoid ongoing cost
- CORS is set to `*` for demo simplicity — a production deployment would restrict this to a specific origin
- IAM roles here are broadly scoped (`*FullAccess` policies) for speed of setup — a production version would tighten these to least-privilege
- Credentials are never committed — see `.gitignore`

---

## 🔮 Possible extensions

- Swap the manually-triggered Glue job for an EventBridge-triggered pipeline on new S3 uploads
- Add a SageMaker Pipeline to orchestrate training → evaluation → conditional deployment
- Add model monitoring / drift detection via SageMaker Model Monitor
- Migrate the frontend to React with proper component structure and state management