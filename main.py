import streamlit as st
import pandas as pd
import pickle


@st.cache_resource
def load_model():
    with open('xgb_model1.pkl', 'rb') as file:
        loaded_xgb_model = pickle.load(file)
    return loaded_xgb_model


# Page Configurations
st.set_page_config(layout="wide", page_title='Heart Disease Prediction')

# Sidebar Information
st.sidebar.title("Heart Disease Prediction Application")
with st.sidebar.expander("About"):
    st.write(
        "This application was built on the Heart Failure Prediction dataset from "
        "[Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)."
        " The source code and documentation can be found on "
        "[GitHub](https://github.com/2003UJAN/2003UJAN-SRM-AI-Sem-4-Project-Heart-Disease-Prediction)."
    )
with st.sidebar.expander("Model Metrics"):
    st.subheader("Decision Tree:")
    st.write("Train = 84.06%, Test = 85.33%")
    st.subheader("Random Forest:")
    st.write("Train = 93.05%, Test = 89.67%")
    st.subheader("XGBoost:")
    st.write("Train = 95.78%, Test = 90.76%")

st.sidebar.write("Parameters can be tuned further for better results.")

# Load Model
xgb_model = load_model()

st.header("Heart Disease Prediction - Decision Trees, Random Forest, and XGBoost")

# Layout
col1, _, col2 = st.columns([1.5, 0.2, 1.5])

# User Inputs - Left Column
with col1:
    age = st.text_input("Age (in years):")
    sex = st.selectbox("Sex:", ["", "Male", "Female"])
    chestPainType = st.selectbox(
        "Chest Pain Type:", ["", "TA", "ATA", "NAP", "ASY"],
        help="TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic"
    )
    restingBP = st.text_input("Resting Blood Pressure [mm Hg]: ")
    cholesterol = st.text_input("Serum Cholesterol [mg/dL]: ")
    fastingBS = st.text_input("Fasting Blood Sugar [mg/dL]: ")

# User Inputs - Right Column
with col2:
    restingECG = st.selectbox(
        "Resting ECG Results:", ["", "Normal", "ST", "LVH"],
        help="ST: ST-T wave abnormality, LVH: Left Ventricular Hypertrophy"
    )
    maxHR = st.text_input("Maximum Heart Rate Achieved: ")
    exerciseAngina = st.selectbox("Exercise-induced Angina:", ["", "Yes", "No"])
    oldpeak = st.text_input("ST Depression (Oldpeak): ")
    st_slope = st.selectbox("ST Slope:", ["", "Up", "Flat", "Down"])

# Prediction Button
if st.button("Run Model"):
    # Input validation
    required_fields = [age, restingBP, cholesterol, fastingBS, maxHR, oldpeak]
    if any(field == '' for field in required_fields):
        st.error("Please fill in all required fields.")
    elif sex == '' or chestPainType == '' or restingECG == '' or exerciseAngina == '' or st_slope == '':
        st.error("Please fill in all dropdown selections.")
    else:
        # Preprocess input data
        data = {
            'Age': [int(age)],
            'RestingBP': [int(restingBP)],
            'Cholesterol': [int(cholesterol)],
            'FastingBS': [0 if int(fastingBS) <= 120 else 1],
            'MaxHR': [int(maxHR)],
            'Oldpeak': [float(oldpeak)],
            'Sex_F': [1 if sex == 'Female' else 0],
            'Sex_M': [1 if sex == 'Male' else 0],
            'ChestPainType_ASY': [1 if chestPainType == "ASY" else 0],
            'ChestPainType_ATA': [1 if chestPainType == "ATA" else 0],
            'ChestPainType_NAP': [1 if chestPainType == "NAP" else 0],
            'ChestPainType_TA': [1 if chestPainType == "TA" else 0],
            'RestingECG_LVH': [1 if restingECG == "LVH" else 0],
            'RestingECG_Normal': [1 if restingECG == "Normal" else 0],
            'RestingECG_ST': [1 if restingECG == "ST" else 0],
            'ExerciseAngina_N': [0 if exerciseAngina == 'Yes' else 1],
            'ExerciseAngina_Y': [0 if exerciseAngina == 'No' else 1],
            'ST_Slope_Down': [1 if st_slope == "Down" else 0],
            'ST_Slope_Flat': [1 if st_slope == "Flat" else 0],
            'ST_Slope_Up': [1 if st_slope == "Up" else 0]
        }

        df = pd.DataFrame(data)

        # Make prediction
        prediction = xgb_model.predict(df.iloc[0:1])

        # Display results
        if prediction[0] == 0:
            st.success("✅ Patient does NOT have heart disease.")
        else:
            st.error("❌ Patient has heart disease.")
