from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st

# Cache model
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi().fetch(video_id)
    text = " ".join([t.text for t in transcript])
    return text

def summarize_text(text):
    # Take first 500 words
    words = text.split()
    text = " ".join(words[:500])
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=250,
        min_length=150,
        do_sample=False
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# UI
st.title("🎥 YouTube Video Summarizer")
st.write("Enter a YouTube video URL to get a summary")

url = st.text_input("YouTube URL:")

if st.button("Summarize"):
    if url.strip() == "":
        st.warning("Please enter a YouTube URL first")
    else:
        try:
            # Extract video ID
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1]
            else:
                st.error("Invalid YouTube URL")
                st.stop()
            
            with st.spinner("Fetching transcript..."):
                text = get_transcript(video_id)
                st.info(f"Transcript: {len(text.split())} words")
            
            with st.spinner("Summarizing..."):
                summary = summarize_text(text)
            
            st.subheader("Summary:")
            st.write(summary)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Make sure video has subtitles/captions enabled")