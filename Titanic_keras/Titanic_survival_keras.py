import streamlit as st
import keras
import pickle
import numpy as np

# Load model and scaler
model = keras.models.load_model('Titanic_model.keras')

with open('titanic_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# UI
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability")

# Input fields
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        help="1 = First, 2 = Second, 3 = Third"
    )
    
    sex = st.selectbox(
        "Sex",
        options=["Male", "Female"]
    )
    
    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=25
    )

with col2:
    sibsp = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0
    )
    
    parch = st.number_input(
        "Parents/Children Aboard",
        min_value=0,
        max_value=6,
        value=0
    )
    
    fare = st.number_input(
        "Fare Paid (£)",
        min_value=0.0,
        max_value=520.0,
        value=32.0
    )

# Predict button
if st.button("Predict Survival"):
    
    # Prepare input — same order as training
    sex_encoded = 1 if sex == "Male" else 0
    
    input_data = np.array([[
        pclass,
        sex_encoded,
        age,
        sibsp,
        parch,
        fare
    ]])
    
    # Scale — same scaler used in training
    input_scaled = scaler.transform(input_data)
    
    # Predict
    probability = model.predict(input_scaled, verbose=0)[0][0]
    
    # Show result
    st.divider()
    
    if probability > 0.5:
        st.success(f"✅ SURVIVED")
        st.metric("Survival Probability", f"{probability:.1%}")
    else:
        st.error(f"❌ DID NOT SURVIVE")
        st.metric("Survival Probability", f"{probability:.1%}")
    
    # Show contributing factors
    st.divider()
    st.subheader("Your Input Summary")
    st.write(f"- Class: {pclass}({'First' if pclass==1 else 'Second' if pclass==2 else 'Third'})")
    st.write(f"- Sex: {sex}")
    st.write(f"- Age: {age}")
    st.write(f"- Family aboard: {sibsp + parch} members")
    st.write(f"- Fare: £{fare}")