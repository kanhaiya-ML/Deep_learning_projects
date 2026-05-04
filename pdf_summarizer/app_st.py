from PyPDF2 import PdfReader
from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

classifier = load_model()

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def summarize(text):
    words = text.split()
    chunks = []
    chunk_size = 500

    for i in range(0, min(len(words), 3000), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    st.info(f"Total chunks: {len(chunks)}")

    chunk_summaries = []
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        summary = classifier(
            chunk,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        chunk_summaries.append(summary[0]['summary_text'])
        progress.progress((i+1) / len(chunks))
        st.caption(f"Chunk {i+1}/{len(chunks)} done")

    final_text = " ".join(chunk_summaries)

    final_summary = classifier(
        final_text,
        max_length=230,
        min_length=50,
        truncation=False,
        do_sample=False
    )

    return final_summary[0]['summary_text']

# UI
st.title("📄 Text & PDF Summarizer")
st.write("Upload a PDF or paste text to get a summary")

# ← TWO TABS
tab1, tab2 = st.tabs(["📄 Upload PDF", "📝 Paste Text"])

# Tab 1 — PDF Upload
with tab1:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        if st.button("Summarize PDF"):
            with st.spinner("Extracting text..."):
                text = extract_text(uploaded_file)
                st.info(f"Total words: {len(text.split())}")

            with st.spinner("Summarizing..."):
                summary = summarize(text)

            st.subheader("Summary:")
            st.write(summary)

            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

# Tab 2 — Paste Text
with tab2:
    pasted_text = st.text_area(
        "Paste your text here:",
        height=300
    )

    st.caption(f"Words: {len(pasted_text.split())} / Characters: {len(pasted_text)}")

    if st.button("Summarize Text"):
        if pasted_text.strip() == "":
            st.warning("Please paste some text first")
        else:
            with st.spinner("Summarizing..."):
                summary = summarize(pasted_text)

            st.subheader("Summary:")
            st.write(summary)

            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )