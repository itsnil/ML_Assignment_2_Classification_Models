import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Setup
st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

# Load Results & Encoders
try:
    results = joblib.load('model_results.pkl')
except:
    st.error("Model results not found. Please run 'train_model.py' first.")
    st.stop()

# Title
st.title("🏥 Diabetes Risk Prediction System")
st.markdown("""
This app predicts the likelihood of early-stage diabetes using 6 different Machine Learning models.
**Dataset**: UCI Diabetes Data Upload.
""")

# --- Sidebar: Data Upload ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])

if uploaded_file:
    st.sidebar.success("File Uploaded Successfully")
    # Logic to process uploaded file would go here
    # For this assignment, we focus on the interactive inputs below

# --- Sidebar: Inputs ---
st.sidebar.header("2. Patient Symptoms")

def user_input_features():
    # We map 'Yes' to 1 and 'No' to 0 because that's how LabelEncoder works (alphabetical)
    def yes_no(label):
        return 1 if st.sidebar.selectbox(label, ('No', 'Yes')) == 'Yes' else 0
    
    age = st.sidebar.number_input("Age", 1, 120, 40)
    gender = 1 if st.sidebar.selectbox("Gender", ('Female', 'Male')) == 'Male' else 0 # F=0, M=1
    
    polyuria = yes_no("Polyuria")
    polydipsia = yes_no("Polydipsia")
    weight_loss = yes_no("Sudden Weight Loss")
    weakness = yes_no("Weakness")
    polyphagia = yes_no("Polyphagia")
    genital_thrush = yes_no("Genital Thrush")
    visual_blurring = yes_no("Visual Blurring")
    itching = yes_no("Itching")
    irritability = yes_no("Irritability")
    delayed_healing = yes_no("Delayed Healing")
    partial_paresis = yes_no("Partial Paresis")
    muscle_stiffness = yes_no("Muscle Stiffness")
    alopecia = yes_no("Alopecia")
    obesity = yes_no("Obesity")

    data = {
        'Age': age, 'Gender': gender, 'Polyuria': polyuria, 'Polydipsia': polydipsia,
        'sudden weight loss': weight_loss, 'weakness': weakness, 'Polyphagia': polyphagia,
        'Genital thrush': genital_thrush, 'visual blurring': visual_blurring, 'Itching': itching,
        'Irritability': irritability, 'delayed healing': delayed_healing, 'partial paresis': partial_paresis,
        'muscle stiffness': muscle_stiffness, 'Alopecia': alopecia, 'Obesity': obesity
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Main Page ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    
    # Model Selection Dropdown
    model_list = ["Logistic Regression", "Decision Tree", "KNN", 
                  "Naive Bayes", "Random Forest", "XGBoost"]
    model_name = st.selectbox("Select Model", model_list)
    
    # Load specific model file
    file_map = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "KNN": "knn_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }
    
    try:
        model = joblib.load(file_map[model_name])
    except:
        st.warning(f"Could not load {model_name}. Make sure train_model.py ran successfully.")
        st.stop()

    if st.button("Predict Risk"):
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)
        
        st.write("---")
        if prediction[0] == 1:
            st.error(f"**Result: Positive (High Risk)**")
            st.write(f"Probability: {prob[0][1]:.2%}")
        else:
            st.success(f"**Result: Negative (Low Risk)**")
            st.write(f"Probability: {prob[0][0]:.2%}")

with col2:
    st.subheader("Model Performance")
    
    # Display Metrics
    m = results[model_name]
    st.write(f"**Metrics for {model_name}:**")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['Accuracy']:.2%}")
    c2.metric("Precision", f"{m['Precision']:.2%}")
    c3.metric("Recall", f"{m['Recall']:.2%}")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("F1 Score", f"{m['F1']:.2f}")
    c5.metric("AUC", f"{m['AUC']:.2f}")
    c6.metric("MCC", f"{m['MCC']:.2f}")
    
    # Display Confusion Matrix
    st.write("---")
    st.write("**Confusion Matrix:**")
    img_name = file_map[model_name].replace("_model.pkl", "_cm.png")
    try:
        image = Image.open(img_name)
        st.image(image, caption=f"Confusion Matrix - {model_name}", use_column_width=True)
    except:
        st.warning("Confusion matrix image not found.")

st.markdown("---")
st.caption("BITS Pilani - Machine Learning Assignment 2")

