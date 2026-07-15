%%writefile src/train.py

import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb

# ---------- Training ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    input_file = [f for f in os.listdir(args.train) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(args.train, input_file))

    categorical_cols = [
        "gender", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall: {recall_score(y_test, preds):.4f}")
    print(f"AUC: {roc_auc_score(y_test, probs):.4f}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(list(X.columns), os.path.join(args.model_dir, "feature_columns.joblib"))

# ---------- Inference (used by the deployed endpoint) ----------
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    return {"model": model, "feature_columns": feature_columns}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        import json
        data = json.loads(request_body)
        return pd.DataFrame(data if isinstance(data, list) else [data])
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    model = model_artifacts["model"]
    feature_columns = model_artifacts["feature_columns"]

    # Align incoming data to the exact columns/order used at training time
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    preds = model.predict(input_data)
    probs = model.predict_proba(input_data)[:, 1]
    return {"predictions": preds.tolist(), "probabilities": probs.tolist()}

def output_fn(prediction, response_content_type):
    import json
    return json.dumps(prediction)

if __name__ == "__main__":
    main()