import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings("ignore")

DATA_PATH  = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.pkl"

def run():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(n_estimators=200, learning_rate=0.05,
                              scale_pos_weight=3, eval_metric="logloss",
                              random_state=42, verbosity=0))
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {auc:.4f}")
    joblib.dump(pipe, MODEL_PATH)
    print(f"\n✅ Model da luu: {MODEL_PATH}")

if __name__ == "__main__":
    run()