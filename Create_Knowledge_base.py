from pdf2image import convert_from_path
import pytesseract
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Improved OCR for Bengali
def extract_bengali_text(pdf_path):
    images = convert_from_path(pdf_path, dpi=350, poppler_path=r'C:\\Users\\Khairul Islam\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin')  # Higher DPI
    all_text = ""
    
    # Custom Tesseract configuration for Bengali
    custom_config = r'--oem 3 --psm 6 -l ben+eng'  # PSM 6: Assume uniform block of text
    
    for i, img in enumerate(images):
        # Preprocess image for better OCR
        img = img.convert('L')  # Grayscale
        img = img.point(lambda x: 0 if x < 140 else 255)  # Thresholding
        
        text = pytesseract.image_to_string(
            img, 
            lang='ben',
            config=custom_config
        )
        all_text += text + "\n"
    
    return all_text

# 2. Enhanced Bengali Text Cleaning
def clean_bangla_text(text):
    # Preserve Bengali characters, numbers, and punctuation
    text = re.sub(r'[^\u0980-\u09FF০-৯।,!?\'"()\s]', '', text)
    
    # Normalize Bengali punctuation
    text = re.sub(r'[।]', '। ', text)  # Add space after full stops
    text = re.sub(r'([,!?])', r' \1 ', text)  # Space around punctuation
    
    # Remove line breaks within sentences
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newline → space
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    
    return text.strip()

# Extract and clean text
pdf_path = "HSC26-Bangla1st-Paper.pdf"
raw_text = extract_bengali_text(pdf_path)
cleaned_text = clean_bangla_text(raw_text)

# Save cleaned text
with open("bangla_cleaned_ocr.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

# 3. Context-Preserving Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,  # Smaller chunks for precise answers
    chunk_overlap=75,  # More overlap for context
    separators=["\n\n", "।", "\n", " ", ""],  # Bengali-specific separators
    length_function=len
)

chunks = text_splitter.split_text(cleaned_text)
print(f"Total chunks: {len(chunks)}")
print("Sample chunk:", chunks[0][:150])

# 4. Better Multilingual Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if available
    encode_kwargs={'normalize_embeddings': True}
)

# 5. Create and save vector store
docs = [Document(page_content=text) for text in chunks]
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local("faiss_index_bangla")

# 6. Load vector store
vectorstore = FAISS.load_local(
    "faiss_index_bangla", 
    embedding_model, 
    allow_dangerous_deserialization=True
)
