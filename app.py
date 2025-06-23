import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Set page config for layout and title
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔍", layout="centered")

# Add custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 1.2em;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with emoji and styling
st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>📉 Customer Churn Prediction</h1>", unsafe_allow_html=True)

st.markdown("### 🧠 Enter customer information below to predict the churn:")

st.divider()

# Input layout in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧓 Age", min_value=10, max_value=100, value=30, help="Customer's age")
    tenure = st.number_input("📅 Tenure (in months)", min_value=0, max_value=130, value=10, help="How long the customer has been with the company")

with col2:
    monthly_charge = st.number_input("💵 Monthly Charges", min_value=30, max_value=150, help="Monthly billing amount")
    gender = st.selectbox("🧍 Gender", ["Male", "Female"])

st.divider()

# Predict button
predictbutton = st.button("🔍 Predict Churn")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0
    X = [age, gender_selected, tenure, monthly_charge]
    X_array = scaler.transform([np.array(X)])
    prediction = model.predict(X_array)[0]
    predicted = "Yes" if prediction == 1 else "No"

    # Result display with effect
    st.success("✅ Prediction Completed!")
    st.write(f"### 🔮 Will the customer churn? → **{predicted}**")
    
    if predicted == "Yes":
        st.warning("⚠️ This customer is likely to churn. Take proactive action.")
    else:
        st.balloons()
        st.info("🎉 This customer is likely to stay!")

else:
    st.info("ℹ️ Enter the values above and click **Predict** to get the result.")
