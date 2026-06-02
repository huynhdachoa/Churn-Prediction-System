"""
train.py — Nâng cấp: feature engineering + tuning + logging + validation
Chạy: python src/train.py
"""
import os, json, logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, precision_score, recall_score)
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings("ignore")

# ── Logging setup ──────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

DATA_PATH  = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.pkl"
META_PATH  = "models/model_metadata.json"

# ── Feature Engineering ────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo thêm features mới có ý nghĩa kinh doanh."""
    df = df.copy()

    # 1. Avg monthly spend per tenure month
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 2. Charge ratio: monthly so với trung bình
    avg_charge = df["MonthlyCharges"].mean()
    df["ChargeRatio"] = df["MonthlyCharges"] / avg_charge

    # 3. Số dịch vụ đang dùng
    service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1
    )

    # 4. Khách hàng mới (tenure < 6 tháng)
    df["IsNewCustomer"] = (df["tenure"] < 6).astype(int)

    # 5. Hợp đồng dài hạn
    df["IsLongTermContract"] = (df["Contract"] == "Two year").astype(int)

    log.info(f"Feature engineering: thêm 5 features mới — shape: {df.shape}")
    return df

# ── Preprocessing ──────────────────────────────────────────
def load_and_preprocess(path: str):
    log.info(f"Loading data: {path}")
    df = pd.read_csv(path)
    log.info(f"Raw data: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isnull().sum()
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    log.info(f"Fixed {n_missing} missing TotalCharges")

    # Drop customerID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Feature engineering
    df = feature_engineering(df)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    churn_rate = df["Churn"].mean()
    log.info(f"Churn rate: {churn_rate:.2%} | Class imbalance ratio: {(1-churn_rate)/churn_rate:.1f}:1")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    log.info(f"Numerical features ({len(num_cols)}): {num_cols}")
    log.info(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Stratified ✅")

    return X_train, X_test, y_train, y_test, preprocessor

# ── Train & Compare Models ─────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, preprocessor):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        log.info(f"\n{'─'*50}")
        log.info(f"Training: {name}")

        pipe = Pipeline([("pre", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # Cross-validation AUC
        cv_auc = cross_val_score(pipe, X_train, y_train,
                                  cv=cv, scoring="roc_auc", n_jobs=-1).mean()

        metrics = {
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "cv_auc":    round(cv_auc, 4),
            "f1_churn":  round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }

        log.info(f"  ROC-AUC (test): {metrics['roc_auc']} | CV-AUC: {metrics['cv_auc']}")
        log.info(f"  F1: {metrics['f1_churn']} | Precision: {metrics['precision']} | Recall: {metrics['recall']}")
        log.info(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

        results[name] = {"pipe": pipe, "metrics": metrics}

    return results

# ── Select Best & Save ─────────────────────────────────────
def select_and_save(results: dict):
    # Chọn theo ROC-AUC test
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best = results[best_name]

    log.info(f"\n{'='*50}")
    log.info(f"🏆 Best model: {best_name}")
    for k, v in best["metrics"].items():
        log.info(f"   {k}: {v}")

    # Lưu model
    joblib.dump(best["pipe"], MODEL_PATH)

    # Lưu metadata
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now().isoformat(),
        "metrics": best["metrics"],
        "all_models": {k: v["metrics"] for k, v in results.items()}
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"✅ Model saved: {MODEL_PATH}")
    log.info(f"✅ Metadata saved: {META_PATH}")

    # In bảng so sánh
    log.info("\n📊 BẢNG SO SÁNH TẤT CẢ MODEL:")
    log.info(f"{'Model':<25} {'AUC':>6} {'CV-AUC':>7} {'F1':>6} {'Recall':>7}")
    log.info("─" * 55)
    for name, res in sorted(results.items(),
                             key=lambda x: x[1]["metrics"]["roc_auc"],
                             reverse=True):
        m = res["metrics"]
        marker = " ← BEST" if name == best_name else ""
        log.info(f"{name:<25} {m['roc_auc']:>6} {m['cv_auc']:>7} {m['f1_churn']:>6} {m['recall']:>7}{marker}")

    return best["pipe"]

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🚀 Bắt đầu training pipeline")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(DATA_PATH)
    results = train_all_models(X_train, X_test, y_train, y_test, preprocessor)
    best_pipe = select_and_save(results)
    log.info("\n✅ Training hoàn tất!")
