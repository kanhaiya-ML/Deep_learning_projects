from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review to predict if it's positive or negative")

user_input = st.text_area("Write Your Review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review first")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]
            label = result["label"]
            score = result["score"]
        
        if label == "POSITIVE":
            st.success(f"POSITIVE 😊 — Confidence: {score:.2%}")
        else:
            st.error(f"NEGATIVE 😞 — Confidence: {score:.2%}")