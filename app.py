import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# --------------------------------------------------------------------------------
# 1. APP CONFIGURATION & SETUP
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide", page_icon="🏥")

# Load saved results and checks
try:
    results = joblib.load('model_results.pkl')
except FileNotFoundError:
    st.error("Critical files not found. Please run 'train_model.ipynb' first to generate models and metrics.")
    st.stop()

# Define file paths for models and images
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

cm_files = {
    "Logistic Regression": "logistic_regression_cm.png",
    "Decision Tree": "decision_tree_cm.png",
    "KNN": "knn_cm.png",
    "Naive Bayes": "naive_bayes_cm.png",
    "Random Forest": "random_forest_cm.png",
    "XGBoost": "xgboost_cm.png"
}

# --------------------------------------------------------------------------------
# 2. SIDEBAR: INPUTS & CONFIGURATION
# --------------------------------------------------------------------------------
st.sidebar.title("Configuration")

# A. Dataset Upload (Requirement a)
st.sidebar.subheader("1. Upload Data (Batch Mode)")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# B. Model Selection (Requirement b)
st.sidebar.subheader("2. Select Model")
selected_model_name = st.sidebar.selectbox("Choose Classifier", list(model_files.keys()))

# Load the selected model
try:
    model = joblib.load(model_files[selected_model_name])
except FileNotFoundError:
    st.sidebar.error(f"Model file for {selected_model_name} not found.")
    st.stop()

# --------------------------------------------------------------------------------
# 3. MAIN APPLICATION TABS
# --------------------------------------------------------------------------------
st.title("🏥 Diabetes Risk Prediction System")
st.markdown(f"**Active Model:** `{selected_model_name}`")

# Create 3 Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Prediction", "📊 Evaluation Metrics", "📂 Dataset View"])

# --- TAB 1: PREDICTION (Single & Batch) ---
with tab1:
    col1, col2 = st.columns([1, 1.5])

    # Left Column: Single Patient Input
    with col1:
        st.subheader("Single Patient Check")
        st.info("Adjust settings in the sidebar (if any) or enter below.")
        
        # Form for inputs
        age = st.number_input("Age", 1, 120, 40)
        gender = st.selectbox("Gender", ["Female", "Male"])
        
        symptoms = ["Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia", 
                    "Genital thrush", "visual blurring", "Itching", "Irritability", "delayed healing", 
                    "partial paresis", "muscle stiffness", "Alopecia", "Obesity"]
        
        input_data = {}
        for sym in symptoms:
            input_data[sym] = st.selectbox(sym, ["No", "Yes"])
        
        # Prepare Data for Prediction (Encoding: Yes=1, No=0, Male=1, Female=0)
        row = [age, 1 if gender == "Male" else 0] + [1 if input_data[sym] == "Yes" else 0 for sym in symptoms]
        input_df = pd.DataFrame([row], columns=['Age', 'Gender'] + symptoms)
        
        if st.button("Predict Risk"):
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            
            st.write("---")
            if prediction == 1:
                st.error(f"**Result: POSITIVE (High Risk)**")
                st.write(f"Confidence: **{prob[1]:.2%}**")
            else:
                st.success(f"**Result: NEGATIVE (Low Risk)**")
                st.write(f"Confidence: **{prob[0]:.2%}**")

    # Right Column: Batch Prediction (Upload)
    with col2:
        st.subheader("Batch Prediction (from CSV)")
        if uploaded_file is not None:
            # Read and show preview
            uploaded_file.seek(0)
            df_batch = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview (First 5 rows):")
            st.dataframe(df_batch.head())
            
            if st.button("Predict on Uploaded Data"):
                try:
                    # Note: This assumes uploaded data is already numerically encoded like the training data.
                    # If real users upload raw text ("Yes"/"No"), you would need to apply encoders here.
                    predictions = model.predict(df_batch)
                    
                    # Append results
                    df_batch['Predicted_Class'] = predictions
                    df_batch['Predicted_Label'] = df_batch['Predicted_Class'].map({1: 'Positive', 0: 'Negative'})
                    
                    st.success("Predictions generated successfully!")
                    st.dataframe(df_batch)
                except Exception as e:
                    st.error(f"Error during prediction. Ensure CSV columns match training features.\nError: {e}")
        else:
            st.info("Upload a CSV file in the sidebar to use this feature.")

# --- TAB 2: EVALUATION METRICS (Requirement c & d) ---
with tab2:
    st.header(f"Performance: {selected_model_name}")
    
    # Get metrics for selected model
    metrics = results.get(selected_model_name, {})
    
    if metrics:
        # Row 1: Primary Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['Precision']:.2%}")
        c3.metric("Recall", f"{metrics['Recall']:.2%}")
        
        # Row 2: Secondary Metrics
        c4, c5, c6 = st.columns(3)
        c4.metric("F1 Score", f"{metrics['F1']:.2f}")
        c5.metric("AUC Score", f"{metrics['AUC']:.2f}")
        c6.metric("MCC Score", f"{metrics['MCC']:.2f}")
    else:
        st.warning("No metrics found for this model.")
        
    st.divider()
    
    # Confusion Matrix Image
    st.subheader("Confusion Matrix")
    try:
        cm_image_path = cm_files.get(selected_model_name)
        image = Image.open(cm_image_path)
        st.image(image, caption=f"Confusion Matrix - {selected_model_name}", width=500)
    except Exception:
        st.warning("Confusion matrix image not found. Ensure 'train_model.ipynb' generated .png files.")

# --- TAB 3: DATASET VIEW (New Tab) ---
with tab3:
    st.header("📂 Dataset Explorer")
    
    if uploaded_file is not None:
        st.success("File Loaded Successfully")
        uploaded_file.seek(0)
        df_view = pd.read_csv(uploaded_file)
        
        st.subheader("Full Dataset")
        st.dataframe(df_view)  # Interactive table
        
        st.subheader("Dataset Statistics")
        st.write(df_view.describe())
    else:
        st.info("Please upload a CSV file in the Sidebar to view it here.")
        
    st.divider()
    st.caption("Developed for BITS Pilani Machine Learning Assignment")
