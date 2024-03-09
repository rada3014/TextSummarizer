
import streamlit as st
import gdown
import shutil
from transformers import pipeline,T5ForConditionalGeneration,T5Tokenizer

# Function to download the model from Google Drive
def download_model_extract():
    # Replace 'YOUR_FILE_ID' with the actual file ID from your Google Drive
    gdown.download("https://drive.google.com/uc?export=download&id=1-Z0kbbOold1wMG1VhILcWEWrGBCBpOHj", "model1.zip", quiet=False)
    shutil.unpack_archive('model1.zip', 'model1', 'zip')


# Download the model if it's not already downloaded
download_model_extract()

def main():
    st.title("Text Summarization")

    # Load the T5-small model
    model = T5ForConditionalGeneration.from_pretrained("model1")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer,min_length=20, max_length=40, truncation=True)

    # User input
    input_text = st.text_area("Enter the text you want to summarize:", height=200)

    # Summarize button
    if st.button("Summarize"):
        if input_text:
            # Generate the summary
            output = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
            summary = output[0]['summary_text']

            # Display the summary as bullet points
            st.subheader("Summary:")
            bullet_points = summary.split(". ")
            for point in bullet_points:
                st.write(f"- {point}")
        else:
            st.warning("Please enter text to summarize.")

if __name__ == "__main__":
    main()
