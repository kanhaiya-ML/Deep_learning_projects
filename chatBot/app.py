from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("🤖 AI Chatbot")
st.write("Powered by DialoGPT")

# Initialize session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Encode input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors='pt'
    )

    # Add to history
    if st.session_state.chat_history_ids is not None:
        input_ids = torch.cat(
            [st.session_state.chat_history_ids, new_input_ids],
            dim=-1
        )
    else:
        input_ids = new_input_ids

    # Generate response
    with st.spinner("Thinking..."):
        st.session_state.chat_history_ids = model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )

    # Decode response
    response = tokenizer.decode(
        st.session_state.chat_history_ids[
            :, input_ids.shape[-1]:
        ][0],
        skip_special_tokens=True
    )

    # Show bot response
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.messages = []
    st.rerun()