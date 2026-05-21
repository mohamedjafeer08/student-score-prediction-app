import streamlit as st
import numpy as np
import joblib  # 1. Import joblib instead of pickle

@st.cache_resource
def load_saved_model():
    # 2. Use joblib to load the model directly
    loaded_model = joblib.load('students_score_prediction_model.pkl')
    return loaded_model

try:
    model = load_saved_model()
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# ==========================================
# 2. STREAMLIT FRONTEND INTERFACE
# ==========================================
st.title("🎓 Student Exam Score Predictor")
st.write("Enter the student's details below to predict their exam performance using your saved Machine Learning model.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Study Inputs")
    hours = st.slider("Hours Studied per day:", min_value=0.0, max_value=12.0, value=5.0, step=0.5)
    
    prep_input = st.radio("Test Preparation Course:", ("None", "Completed"))
    # Match the encoding mapping you used during model training
    prep = 1 if prep_input == "Completed" else 0

with col2:
    st.subheader("🏡 Background Inputs")
    edu_input = st.selectbox(
        "Highest Parental Education Level:",
        ("High School", "Some College", "Bachelor's Degree", "Master's Degree")
    )
    # Match the encoding mapping you used during model training
    edu_map = {"High School": 1, "Some College": 2, "Bachelor's Degree": 3, "Master's Degree": 4}
    edu = edu_map[edu_input]

st.divider()

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if st.button("Predict Exam Score", type="primary"):
    # Format the inputs into a 2D array matching your training features: [Hours, Prep, Edu]
    input_data = np.array([[hours, prep, edu]])
    
    # Make the prediction using your loaded pickle model
    prediction = model.predict(input_data)[0]
    
    # Optional boundary clip (0-100)
    final_score = np.clip(prediction, 0, 100)
    
    # Display the result
    st.balloons() 
    st.success(f"### 🎯 Predicted Exam Score: **{final_score:.1f} / 100**")