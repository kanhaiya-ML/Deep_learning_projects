from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model='gpt2'
    )

generator = load_model()

# UI
st.title("✍️ AI Text Generator")
st.write("Powered by GPT-2")

user_input = st.text_input("Enter your topic or starting sentence:")

max_length = st.slider(
    "Output length (words)",
    min_value=50,
    max_value=200,
    value=100
)

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter something first")
    else:
        with st.spinner("Generating..."):
            result = generator(
                user_input,
                max_length=max_length,
                num_return_sequences=1,
                truncation=True
            )
            generated = result[0]['generated_text']
        
        st.subheader("Generated Text:")
        st.write(generated)