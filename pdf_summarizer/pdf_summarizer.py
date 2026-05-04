from PyPDF2 import PdfReader
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="select pdf file",
    filetypes=[("pdf files","*pdf")]
)

reader = PdfReader(file_path)
text = ""
for page in reader.pages:
    text += page.extract_text()

classifier = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

words = text.split()
chunks = []
chunk_size = 500

for i in range(0,min(len(words),3000),chunk_size):
    chunk = " ".join(words[i:i+chunk_size])
    chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")

chunk_summarize = []
for i,chunk in enumerate(chunks):
    summary = classifier(
        chunk,
        max_length=150,
        min_length=50,
        do_sample=False,
    )
    chunk_summarize.append(summary[0]["summary_text"])
    print(f"Chunk {i+1} Done")

final_text = " ".join(chunk_summarize)

final_summary = classifier(
    final_text,
    max_length=230,
    min_length=50,
    truncation=True
)
print("Final Summary")
print(f"Final Summary: {final_summary[0]["summary_text"]}")


