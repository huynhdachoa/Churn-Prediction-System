import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

model = load_model()

st.title("📊 Customer Churn Prediction")
st.markdown("Nhập thông tin khách hàng để dự đoán nguy cơ rời bỏ dịch vụ")
st.divider()

st.sidebar.header("Thông tin khách hàng")
gender        = st.sidebar.selectbox("Giới tính", ["Male", "Female"])
senior        = st.sidebar.selectbox("Khách cao tuổi", [0, 1])
partner       = st.sidebar.selectbox("Có partner", ["Yes", "No"])
dependents    = st.sidebar.selectbox("Có người phụ thuộc", ["Yes", "No"])
tenure        = st.sidebar.slider("Số tháng sử dụng", 0, 72, 12)
phone         = st.sidebar.selectbox("Dịch vụ điện thoại", ["Yes", "No"])
multi_lines   = st.sidebar.selectbox("Nhiều đường dây", ["Yes", "No", "No phone service"])
internet      = st.sidebar.selectbox("Dịch vụ Internet", ["Fiber optic", "DSL", "No"])
online_sec    = st.sidebar.selectbox("Bảo mật trực tuyến", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Sao lưu trực tuyến", ["Yes", "No", "No internet service"])
device_prot   = st.sidebar.selectbox("Bảo vệ thiết bị", ["Yes", "No", "No internet service"])
tech_support  = st.sidebar.selectbox("Hỗ trợ kỹ thuật", ["Yes", "No", "No internet service"])
streaming_tv  = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_mv  = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract      = st.sidebar.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
paperless     = st.sidebar.selectbox("Hóa đơn điện tử", ["Yes", "No"])
payment       = st.sidebar.selectbox("Phương thức thanh toán", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
monthly       = st.sidebar.number_input("Phí hàng tháng (USD)", 0.0, 200.0, 65.0)
total         = st.sidebar.number_input("Tổng phí đã trả (USD)", 0.0, 10000.0, monthly * tenure)

if st.sidebar.button("🔍 Dự đoán", use_container_width=True):
    sample = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_backup,
        "DeviceProtection": device_prot, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total
    }])

    prob = model.predict_proba(sample)[0, 1]
    pred = prob >= 0.5

    col1, col2, col3 = st.columns(3)
    col1.metric("Xác suất Churn", f"{prob:.1%}")
    col2.metric("Kết quả", "⚠️ CHURN" if pred else "✅ STAY")
    col3.metric("Rủi ro", "🔴 Cao" if prob > 0.7 else "🟡 Trung bình" if prob > 0.4 else "🟢 Thấp")

    if pred:
        st.error(f"⚠️ Khách hàng có nguy cơ rời bỏ ({prob:.1%}). Cần can thiệp sớm!")
    else:
        st.success(f"✅ Khách hàng có khả năng tiếp tục sử dụng ({1-prob:.1%}).")

    st.subheader("🔎 Giải thích dự đoán (SHAP)")
    X_t = model.named_steps["pre"].transform(sample)
    feat_names = model.named_steps["pre"].get_feature_names_out().tolist()
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    sv = explainer.shap_values(X_t)
    sv1 = sv[1] if isinstance(sv, list) else sv
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv1[0],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_t[0], feature_names=feat_names
        ), show=False)
    st.pyplot(fig)
else:
    st.info("👈 Nhập thông tin ở sidebar rồi bấm Dự đoán")