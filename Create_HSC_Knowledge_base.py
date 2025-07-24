from pdf2image import convert_from_path
import re
import os
import numpy as np
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# 1. Extract text from HSC PDF with limited pages for testing
def extract_hsc_text(pdf_path, max_pages=10):
    """
    Extract text from HSC PDF with PaddleOCR
    max_pages: Limit processing to first N pages for testing
    """
    print(f"🔄 Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')  # Updated parameter
    
    # Convert PDF to images
    print(f"🔄 Converting PDF to images...")
    images = convert_from_path(
        pdf_path, 
        dpi=300  # Reduced DPI for faster processing
        # poppler_path is not needed on Linux if poppler-utils is installed
    )
    
    # Limit pages for initial testing
    images = images[:max_pages]
    
    all_text = ""
    print(f"📄 Processing {len(images)} pages...")
    
    for i, img in enumerate(images):
        print(f"   Processing page {i+1}/{len(images)}...")
        
        try:
            # Convert PIL image to numpy array
            img_np = np.array(img)
            
            # Process image with PaddleOCR
            result = ocr.predict(img_np)  # Using predict instead of ocr
            
            # Extract text from results
            page_text = ""
            if result and len(result) > 0:
                for line in result:
                    if line and len(line) >= 2 and len(line[1]) >= 2:
                        text_content = line[1][0]  # Extract text
                        confidence = line[1][1]   # Extract confidence
                        if confidence > 0.5:  # Only include high-confidence text
                            page_text += text_content + " "
            
            all_text += page_text + "\n\n"
            print(f"   ✅ Page {i+1} processed ({len(page_text)} characters)")
            
        except Exception as e:
            print(f"   ⚠️ Error processing page {i+1}: {e}")
            continue
    
    return all_text

# 2. Clean HSC text for educational content
def clean_hsc_text(text):
    """
    Clean extracted text for HSC educational content
    """
    # Preserve Bengali, English, numbers, and educational symbols
    text = re.sub(r'[^\u0980-\u09FF০-৯a-zA-Z0-9।,!?\'"()\s.-:;]', '', text)
    
    # Normalize spacing and punctuation
    text = re.sub(r'[।]', '। ', text)
    text = re.sub(r'([,!?:;])', r' \1 ', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short lines (OCR artifacts)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 5]
    
    return '\n'.join(cleaned_lines).strip()

# 3. Main processing function
def create_hsc_knowledge_base(max_pages=10):
    """
    Create knowledge base from HSC PDF
    """
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    
    print(f"🎯 Creating HSC Knowledge Base (processing {max_pages} pages)")
    
    # Extract and clean text
    print("🔄 Extracting text from HSC PDF...")
    raw_text = extract_hsc_text(pdf_path, max_pages)
    
    print("🔄 Cleaning extracted text...")
    cleaned_text = clean_hsc_text(raw_text)
    
    # Save cleaned text
    output_file = f"hsc_bangla_cleaned_ocr_{max_pages}pages.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    print(f"✅ Cleaned text saved to {output_file}")
    print(f"📊 Total characters: {len(cleaned_text)}")
    
    # Create chunks
    print("🔄 Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Larger chunks for educational content
        chunk_overlap=100,
        separators=["\n\n", "।", "\n", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_text(cleaned_text)
    print(f"📝 Created {len(chunks)} chunks")
    
    if chunks:
        print(f"📖 Sample chunk:")
        print("-" * 50)
        print(chunks[0][:300] + "..." if len(chunks[0]) > 300 else chunks[0])
        print("-" * 50)
    
    # Create embeddings
    print("🔄 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store
    print("🔄 Creating vector store...")
    docs = [Document(
        page_content=text, 
        metadata={
            "source": "HSC26-Bangla1st-Paper", 
            "chunk_id": i,
            "pages_processed": max_pages
        }
    ) for i, text in enumerate(chunks)]
    
    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    # Save vector store
    index_name = f"faiss_index_hsc_bangla_{max_pages}pages"
    vectorstore.save_local(index_name)
    print(f"✅ Vector store saved as '{index_name}'")
    
    # Test the vector store
    print("🔍 Testing vector store...")
    test_queries = [
        "বাংলা সাহিত্য",
        "কবিতা",
        "গল্প",
        "ব্যাকরণ"
    ]
    
    for query in test_queries:
        relevant_docs = vectorstore.similarity_search(query, k=2)
        print(f"📋 Query: '{query}' -> Found {len(relevant_docs)} docs")
        if relevant_docs:
            print(f"   Preview: {relevant_docs[0].page_content[:100]}...")
    
    print(f"\n🎉 HSC Knowledge Base created successfully!")
    print(f"📁 Files created:")
    print(f"   - {output_file}")
    print(f"   - {index_name}/")
    
    return index_name, output_file

# Run the function
if __name__ == "__main__":
    # Start with just 5 pages for testing
    create_hsc_knowledge_base(max_pages=5)
