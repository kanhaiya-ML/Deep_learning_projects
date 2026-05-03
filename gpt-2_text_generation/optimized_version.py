from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model(model_name):
    return pipeline("text-generation", model=model_name)

st.title("✍️ GPT Text Generator")
st.write("Powered by GPT-2")

# Model selection
model_choice = st.selectbox(
    "Choose Model:",
    options={
        "GPT-2 Small (Fast)": "gpt2",
        "DistilGPT-2 (Tiny)": "distilgpt2"
    }.keys()
)

models = {
    "GPT-2 Small (Fast)": "gpt2",
    "DistilGPT-2 (Tiny)": "distilgpt2"
}


styles = {
    "None":          "",
    "Story":         "Once upon a time, ",
    "News Article":  "Breaking News: ",
    "Formal":        "It is hereby stated that ",
    "Poem":          "Roses are red, violets are blue, ",
    "Motivational":  "Never give up because "
}

style_choice = st.selectbox(
    "Writing Style:",
    ["None", "Story", "News Article", "Formal", "Poem", "Motivational"]
)


selected_model = models[model_choice]
generator = load_model(selected_model)
# Style selection
# User input
user_input = st.text_area("Enter Your Sentence:")

# Parameters
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider(
        "Temperature (creativity)",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1
    )
    max_length = st.slider(
        "Output Length",
        min_value=50,
        max_value=300,
        value=150
    )

with col2:
    num_sequences = st.slider(
        "Number of Versions",
        min_value=1,
        max_value=3,
        value=1
    )
    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.3,
        step=0.1
    )

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence first")
    else:
        # Add style prefix
        selected_style = styles[style_choice]
        prompt = selected_style + user_input

        with st.spinner(f"Generating with {model_choice}..."):
            results = generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_sequences,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                truncation=True
            )

        for i, result in enumerate(results):
            st.subheader(f"Version {i+1}")
            
            # Remove input from output
            generated_only = result['generated_text'][len(user_input):]

            st.code(generated_only)  # adds copy button automatically
            
            st.write(generated_only)
            st.caption(f"Word count: {len(generated_only.split())} words")
            st.divider()