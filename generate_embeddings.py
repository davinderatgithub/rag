import PyPDF2
import torch
from transformers import pipeline
def generate_embeddings(tokenizer, model, device, text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate the model outputs without tracking gradients
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract the final hidden state and compute the mean
    final_hidden_state = outputs.hidden_states[-1].mean(dim=1).tolist()
    
    return text, final_hidden_state

def read_pdf_file(pdf_path):
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_path)

    # Initialize an empty list to store the text lines
    text_lines = []

    # Iterate through each page in the PDF
    for page_num in range(len(pdf_reader.pages)):
        # Extract the text from the current page
        page_text = pdf_reader.pages[page_num].extract_text()

        # Split the page text into individual lines
        page_lines = page_text.splitlines()

        # Add the lines to the text_lines list
        text_lines.extend(page_lines)

    # Return the list of text lines
    return text_lines
