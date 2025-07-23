from pdf2image import convert_from_path
import re
import os
import numpy as np
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def extract_hsc_pdf_batch(pdf_path, start_page=1, end_page=10):
    """
    Extract text from HSC PDF in batches to avoid memory issues
    """
    print(f"🔄 Processing pages {start_page} to {end_page}")
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
    
    # Convert specific pages
    images = convert_from_path(
        pdf_path, 
        dpi=250,  # Reduced DPI for faster processing
        first_page=start_page,
        last_page=end_page,
        poppler_path=r'C:\Users\Khairul Islam\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'
    )
    
    all_text = ""
    for i, img in enumerate(images):
        page_num = start_page + i
        print(f"   Processing page {page_num}...")
        
        try:
            # Convert to numpy array
            img_np = np.array(img)
            
            # Process with OCR
            result = ocr.predict(img_np)
            
            # Extract text
            page_text = ""
            if result and len(result) > 0:
                for line in result:
                    if len(line) >= 2 and len(line[1]) >= 2:
                        text_content = line[1][0]
                        confidence = line[1][1]
                        if confidence > 0.6:  # Higher confidence threshold
                            page_text += text_content + " "
            
            all_text += f"\n\n--- Page {page_num} ---\n" + page_text
            
        except Exception as e:
            print(f"   ⚠️ Error on page {page_num}: {e}")
            continue
    
    return all_text

def process_hsc_pdf_in_batches():
    """
    Process the entire HSC PDF in manageable batches
    """
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    
    print("📚 HSC PDF Batch Processor")
    print("=" * 50)
    
    # Get total pages (approximate)
    total_pages = 49  # Known from previous attempt
    batch_size = 5    # Process 5 pages at a time
    
    print(f"📄 Total pages: {total_pages}")
    print(f"🔄 Batch size: {batch_size}")
    
    all_extracted_text = ""
    
    # Process in batches
    for start_page in range(1, total_pages + 1, batch_size):
        end_page = min(start_page + batch_size - 1, total_pages)
        
        try:
            batch_text = extract_hsc_pdf_batch(pdf_path, start_page, end_page)
            all_extracted_text += batch_text
            
            print(f"✅ Batch {start_page}-{end_page} completed")
            
            # Save progress after each batch
            with open(f"hsc_batch_{start_page}_{end_page}.txt", "w", encoding="utf-8") as f:
                f.write(batch_text)
            
        except Exception as e:
            print(f"❌ Error processing batch {start_page}-{end_page}: {e}")
            continue
        
        # Ask if user wants to continue
        if start_page > 1:
            continue_choice = input(f"\nContinue with next batch? (y/n): ").lower()
            if continue_choice != 'y':
                break
    
    # Clean and save complete text
    if all_extracted_text:
        cleaned_text = clean_extracted_text(all_extracted_text)
        
        final_file = "hsc_complete_ocr.txt"
        with open(final_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        
        print(f"\n✅ Complete text saved to {final_file}")
        print(f"📊 Total characters: {len(cleaned_text)}")
        
        # Create knowledge base
        create_kb = input("\nCreate knowledge base now? (y/n): ").lower()
        if create_kb == 'y':
            create_knowledge_base_from_ocr(cleaned_text)
    
    else:
        print("❌ No text extracted!")

def clean_extracted_text(text):
    """
    Clean OCR extracted text
    """
    # Remove page markers
    text = re.sub(r'\n--- Page \d+ ---\n', '\n\n', text)
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove very short lines (likely OCR errors)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 5]
    
    return '\n'.join(cleaned_lines).strip()

def create_knowledge_base_from_ocr(content):
    """
    Create knowledge base from OCR content
    """
    print("\n🔄 Creating knowledge base from OCR content...")
    
    # Create chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "।", "\n", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_text(content)
    print(f"📝 Created {len(chunks)} chunks")
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create documents
    docs = [Document(
        page_content=chunk, 
        metadata={
            "source": "HSC_OCR_Complete", 
            "chunk_id": i,
            "method": "PaddleOCR_Batch"
        }
    ) for i, chunk in enumerate(chunks)]
    
    # Create vector store
    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    # Save vector store
    index_name = "faiss_index_hsc_ocr_complete"
    vectorstore.save_local(index_name)
    print(f"✅ Knowledge base saved as '{index_name}'")

if __name__ == "__main__":
    process_hsc_pdf_in_batches()
