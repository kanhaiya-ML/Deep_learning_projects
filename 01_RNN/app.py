import streamlit as st
import keras
import pickle
import numpy as np

# Load model and word_index
model = keras.models.load_model('Sentiment_analyzer_model.keras')

with open('word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

# Your predict function
def predict_sentiment(review):
    words = review.lower().split()
    encoded = []
    for w in words:
        idx = word_index.get(w, 2)
        if idx >= 10000:
            idx = 2
        encoded.append(idx)
    
    encoded = [min(i, 9999) for i in encoded]
    
    padded = keras.preprocessing.sequence.pad_sequences(
        [encoded], maxlen=200
    )
    
    score = model.predict(padded, verbose=0)[0][0]
    return score

# UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and I'll predict if it's positive or negative")

review = st.text_area("Your review here:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first")
    else:
        score = predict_sentiment(review)
        if score > 0.5:
            st.success(f"POSITIVE 😊 — Confidence: {score:.2%}")
        else:
            st.error(f"NEGATIVE 😞 — Confidence: {(1-score):.2%}")