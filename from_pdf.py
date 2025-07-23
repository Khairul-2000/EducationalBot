from pdf2image import convert_from_path
import re
import os
import numpy as np
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


# 3. Enhanced Bengali Text Cleaning (same as before)
def clean_bangla_text(text):
    text = re.sub(r'[^\u0980-\u09FF০-৯।,!?\'"()\s]', '', text)
    text = re.sub(r'[।]', '। ', text)
    text = re.sub(r'([,!?])', r' \1 ', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Main processing
pdf_path = "HSC26-Bangla1st-Paper.pdf"

# Extract and clean text
print("Extracting text from PDF...")
raw_text = extract_bengali_text(pdf_path)
print("Cleaning text...")
cleaned_text = clean_bangla_text(raw_text)

# Save cleaned text
with open("bangla_cleaned_ocr.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

# 4. Text chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "।", "\n", " ", ""],
    length_function=len
)

chunks = text_splitter.split_text(cleaned_text)
print(f"Created {len(chunks)} text chunks")

# 5. Embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Create and save vector store
docs = [Document(page_content=chunk) for chunk in chunks]
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local("faiss_index_bangla")
print("Vector store created successfully!")