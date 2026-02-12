import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("breast_cancer_experiment")

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model configurations 
model_configs = [
    {
        "model_name": "RandomForest",
        "model": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    },
    {
        "model_name": "RandomForest",
        "model": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    },
    {
        "model_name": "LogisticRegression",
        "model": LogisticRegression(C=1.0, max_iter=1000)
    },
    {
        "model_name": "LogisticRegression",
        "model": LogisticRegression(C=0.1, max_iter=1000)
    }
]

for config in model_configs:

    with mlflow.start_run():

        model = config["model"]
        model_name = config["model_name"]

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        # Log model type
        mlflow.log_param("model_type", model_name)

        # Log hyperparameters automatically
        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        # Log metric
        mlflow.log_metric("auc", auc)

        # Log model artifact
        mlflow.sklearn.log_model(model, name="model")

        print(f"{model_name} AUC: {auc}")
