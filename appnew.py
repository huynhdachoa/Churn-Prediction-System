
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS tùy chỉnh ──────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #378ADD;
    margin-bottom: 8px;
}
.churn-high   { border-left-color: #E24B4A !important; }
.churn-medium { border-left-color: #EF9F27 !important; }
.churn-low    { border-left-color: #1D9E75 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Load model & metadata ──────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    try:
        with open("models/model_metadata.json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}
    return model, meta

model, meta = load_artifacts()

# ── Header ─────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📊 Customer Churn Prediction System")
    st.caption("Dự đoán và phân tích nguy cơ khách hàng rời bỏ dịch vụ viễn thông")
with col_h2:
    if meta:
        st.metric("Model", meta.get("model_name", "XGBoost"))
        st.metric("ROC-AUC", meta.get("metrics", {}).get("roc_auc", "—"))

st.divider()

# ── Sidebar ─────────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.header("🧑 Thông tin khách hàng")
    with st.sidebar.expander("👤 Thông tin cá nhân", expanded=True):
        gender     = st.selectbox("Giới tính", ["Male", "Female"])
        senior     = st.selectbox("Khách cao tuổi (≥65)", [0, 1], format_func=lambda x: "Có" if x else "Không")
        partner    = st.selectbox("Có partner", ["Yes", "No"], format_func=lambda x: "Có" if x == "Yes" else "Không")
        dependents = st.selectbox("Có người phụ thuộc", ["Yes", "No"], format_func=lambda x: "Có" if x == "Yes" else "Không")
        tenure     = st.slider("Số tháng sử dụng dịch vụ", 0, 72, 12)

    with st.sidebar.expander("📱 Dịch vụ sử dụng", expanded=True):
        phone      = st.selectbox("Dịch vụ điện thoại", ["Yes", "No"])
        multi      = st.selectbox("Nhiều đường dây", ["Yes", "No", "No phone service"])
        internet   = st.selectbox("Dịch vụ Internet", ["Fiber optic", "DSL", "No"])
        online_sec = st.selectbox("Bảo mật trực tuyến", ["Yes", "No", "No internet service"])
        backup     = st.selectbox("Sao lưu trực tuyến", ["Yes", "No", "No internet service"])
        device     = st.selectbox("Bảo vệ thiết bị", ["Yes", "No", "No internet service"])
        support    = st.selectbox("Hỗ trợ kỹ thuật", ["Yes", "No", "No internet service"])
        tv         = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies     = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with st.sidebar.expander("💳 Hợp đồng & Thanh toán", expanded=True):
        contract   = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
        paperless  = st.selectbox("Hóa đơn điện tử", ["Yes", "No"])
        payment    = st.selectbox("Phương thức thanh toán", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly    = st.number_input("Phí hàng tháng (USD)", 0.0, 200.0, 65.0, step=0.5)
        total      = st.number_input("Tổng phí đã trả (USD)", 0.0, 10000.0,
                                      float(monthly * tenure), step=10.0)

    return {
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total
    }

# ── Feature Engineering (khớp với train.py) ───────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeRatio"] = df["MonthlyCharges"] / 64.76  # mean từ training
    service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1)
    df["IsNewCustomer"] = (df["tenure"] < 6).astype(int)
    df["IsLongTermContract"] = (df["Contract"] == "Two year").astype(int)
    return df

# ── Predict function ───────────────────────────────────────
def predict(sample_dict: dict):
    df = pd.DataFrame([sample_dict])
    df = engineer_features(df)
    prob = model.predict_proba(df)[0, 1]
    return prob, df

def risk_label(prob):
    if prob >= 0.7: return "🔴 Cao", "churn-high"
    if prob >= 0.4: return "🟡 Trung bình", "churn-medium"
    return "🟢 Thấp", "churn-low"

# ── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Dự đoán đơn lẻ", "📁 Batch Predict (CSV)", "📈 Dashboard model"])

# ═══════════════════════════════════════════
# TAB 1 — DỰ ĐOÁN ĐƠN LẺ
# ═══════════════════════════════════════════
with tab1:
    inputs = sidebar_inputs()
    predict_btn = st.sidebar.button("🔍 Dự đoán ngay", use_container_width=True, type="primary")

    if predict_btn:
        prob, sample_df = predict(inputs)
        risk, css_class = risk_label(prob)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Xác suất Churn", f"{prob:.1%}")
        c2.metric("Kết quả", "⚠️ CHURN" if prob >= 0.5 else "✅ STAY")
        c3.metric("Mức rủi ro", risk)
        c4.metric("Tin cậy", f"{max(prob, 1-prob):.1%}")

        # Alert box
        if prob >= 0.5:
            st.error(f"⚠️ Khách hàng này có nguy cơ **rời bỏ dịch vụ** ({prob:.1%}). Đề xuất: liên hệ ưu đãi giữ chân ngay!")
        else:
            st.success(f"✅ Khách hàng này có khả năng **tiếp tục sử dụng** ({1-prob:.1%}).")

        # Khuyến nghị
        st.subheader("💡 Khuyến nghị hành động")
        recs = []
        if inputs["Contract"] == "Month-to-month":
            recs.append("📋 Khuyến khích ký hợp đồng dài hạn (1-2 năm) với ưu đãi giảm giá")
        if inputs["tenure"] < 6:
            recs.append("🎁 Chương trình onboarding đặc biệt cho khách hàng mới")
        if inputs["MonthlyCharges"] > 70:
            recs.append("💰 Xem xét gói cước phù hợp hơn hoặc ưu đãi giảm phí")
        if inputs["OnlineSecurity"] == "No":
            recs.append("🔒 Giới thiệu gói bảo mật trực tuyến miễn phí 3 tháng")
        if not recs:
            recs.append("✅ Duy trì chất lượng dịch vụ hiện tại")
        for r in recs:
            st.info(r)

    else:
        st.info("👈 Nhập thông tin khách hàng ở sidebar rồi bấm **Dự đoán ngay**")
        st.markdown("""
        ### Hướng dẫn sử dụng
        1. **Điền thông tin** khách hàng ở sidebar bên trái
        2. **Bấm Dự đoán** để nhận kết quả
        3. **Đọc biểu đồ SHAP** để hiểu tại sao model đưa ra kết quả đó
        4. **Xem khuyến nghị** hành động cụ thể cho khách hàng này
        """)

# ═══════════════════════════════════════════
# TAB 2 — BATCH PREDICT
# ═══════════════════════════════════════════
with tab2:
    st.header("📁 Dự đoán hàng loạt từ file CSV")
    st.markdown("Upload file CSV chứa danh sách khách hàng, hệ thống sẽ dự đoán toàn bộ cùng lúc.")

    # Template download
    template_cols = ["gender","SeniorCitizen","Partner","Dependents","tenure",
                     "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                     "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                     "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
                     "MonthlyCharges","TotalCharges"]
    template_data = {
        "gender": ["Male","Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes","No"],
        "Dependents": ["No","Yes"],
        "tenure": [2, 36],
        "PhoneService": ["Yes","Yes"],
        "MultipleLines": ["No","Yes"],
        "InternetService": ["Fiber optic","DSL"],
        "OnlineSecurity": ["No","Yes"],
        "OnlineBackup": ["No","Yes"],
        "DeviceProtection": ["No","Yes"],
        "TechSupport": ["No","Yes"],
        "StreamingTV": ["No","Yes"],
        "StreamingMovies": ["No","No"],
        "Contract": ["Month-to-month","Two year"],
        "PaperlessBilling": ["Yes","No"],
        "PaymentMethod": ["Electronic check","Bank transfer (automatic)"],
        "MonthlyCharges": [70.35, 45.20],
        "TotalCharges": [151.65, 1627.20]
    }
    template_df = pd.DataFrame(template_data)
    st.download_button(
        "⬇️ Tải file template CSV",
        template_df.to_csv(index=False).encode("utf-8"),
        "template_churn.csv", "text/csv"
    )

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            numeric_cols = ["seniorcitizen", "tenure", "MonthlyCharges", "TotalCharges"]
            for col in numeric_cols:
                if col in batch_df.columns:
                    batch_df[col] = pd.to_numeric(batch_df[col], errors="coerce").fillna(0) 
            
            st.success(f"✅ Đã load {len(batch_df):,} khách hàng")

            # Validate columns
            missing_cols = [c for c in template_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Thiếu cột: {missing_cols}")
            else:
                batch_fe = engineer_features(batch_df)
                probs = model.predict_proba(batch_fe)[:, 1]
                batch_df["ChurnProbability"] = probs.round(4)
                batch_df["ChurnPrediction"] = ["CHURN" if p >= 0.5 else "STAY" for p in probs]
                batch_df["RiskLevel"] = pd.cut(probs,
                    bins=[0, 0.4, 0.7, 1.0],
                    labels=["🟢 Thấp", "🟡 Trung bình", "🔴 Cao"])

                # Tổng quan
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tổng khách hàng", f"{len(batch_df):,}")
                col2.metric("Dự đoán Churn", f"{(probs>=0.5).sum():,}")
                col3.metric("Tỉ lệ Churn", f"{(probs>=0.5).mean():.1%}")
                col4.metric("Rủi ro cao (>70%)", f"{(probs>=0.7).sum():,}")

                # Phân phối rủi ro
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                risk_counts = batch_df["RiskLevel"].value_counts()
                colors = ["#1D9E75", "#EF9F27", "#E24B4A"]
                ax1.pie(risk_counts, labels=risk_counts.index,
                        autopct="%1.1f%%", colors=colors[:len(risk_counts)])
                ax1.set_title("Phân bố mức rủi ro")

                ax2.hist(probs, bins=20, color="#378ADD", edgecolor="white")
                ax2.axvline(0.5, color="red", linestyle="--", label="Ngưỡng 0.5")
                ax2.set_xlabel("Xác suất Churn")
                ax2.set_ylabel("Số khách hàng")
                ax2.set_title("Phân phối xác suất Churn")
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig)

                # Bảng kết quả
                st.subheader("📋 Kết quả chi tiết")
                display_cols = ["tenure", "Contract", "MonthlyCharges",
                                "ChurnProbability", "ChurnPrediction", "RiskLevel"]
                display_cols = [c for c in display_cols if c in batch_df.columns]
                st.dataframe(
                    batch_df[display_cols].sort_values("ChurnProbability", ascending=False),
                    use_container_width=True, height=350
                )

                # Download kết quả
                st.download_button(
                    "⬇️ Tải kết quả CSV",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    "churn_predictions.csv", "text/csv"
                )
        except Exception as e:
            st.error(f"Lỗi: {e}")

# ═══════════════════════════════════════════
# TAB 3 — DASHBOARD MODEL
# ═══════════════════════════════════════════
with tab3:
    st.header("📈 Dashboard hiệu năng model")

    if meta:
        # Model info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ℹ️ Thông tin model")
            st.markdown(f"**Model đang dùng:** {meta.get('model_name', '—')}")
            st.markdown(f"**Trained at:** {meta.get('trained_at', '—')[:19].replace('T',' ')}")
            m = meta.get("metrics", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("ROC-AUC", m.get("roc_auc","—"))
            c2.metric("F1 (Churn)", m.get("f1_churn","—"))
            c3.metric("Recall", m.get("recall","—"))

        with col2:
            st.subheader("📊 So sánh tất cả model")
            all_models = meta.get("all_models", {})
            if all_models:
                comp_df = pd.DataFrame(all_models).T.reset_index()
                comp_df.columns = ["Model", "ROC-AUC", "CV-AUC", "F1", "Precision", "Recall"]
                comp_df = comp_df.sort_values("ROC-AUC", ascending=False)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Bar chart so sánh
        if all_models:
            st.subheader("📊 Biểu đồ so sánh ROC-AUC")
            fig, ax = plt.subplots(figsize=(8, 4))
            names = list(all_models.keys())
            aucs  = [all_models[n]["roc_auc"] for n in names]
            colors_bar = ["#1D9E75" if n == meta.get("model_name") else "#378ADD" for n in names]
            bars = ax.barh(names, aucs, color=colors_bar, edgecolor="white")
            ax.set_xlim(0.7, 1.0)
            ax.set_xlabel("ROC-AUC")
            ax.set_title("So sánh ROC-AUC các model")
            for bar, val in zip(bars, aucs):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va="center", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("⚠️ Chưa có metadata. Chạy `python src/train.py` trước!")

    # Hướng dẫn interpret
    st.subheader("📖 Hướng dẫn đọc kết quả")
    with st.expander("Các metric có nghĩa là gì?"):
        st.markdown("""
        | Metric | Ý nghĩa | Giá trị tốt |
        |--------|----------|-------------|
        | **ROC-AUC** | Khả năng phân biệt Churn/No Churn | > 0.85 |
        | **Recall (Churn)** | % khách hàng churn được phát hiện | Càng cao càng tốt |
        | **Precision (Churn)** | % dự đoán Churn là đúng | Cân bằng với Recall |
        | **F1-score** | Trung bình điều hòa Precision & Recall | > 0.65 |
        | **CV-AUC** | AUC trên 5-fold cross-validation | Gần với test AUC |

        > **Trong bài toán churn:** Recall quan trọng hơn Precision vì bỏ sót khách hàng churn (False Negative) tốn kém hơn cảnh báo nhầm (False Positive).
        """)
