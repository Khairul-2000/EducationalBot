# Create_Knowledge_base_pdf.py - Optimized for Large Files

import re
import os
import gc
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


def clean_bangla_text(text):
    """Enhanced Bengali text cleaning with memory optimization"""
    print("🧹 Cleaning Bengali text...")
    
    # Remove unwanted characters but keep Bengali, English, numbers
    text = re.sub(r'[^\u0980-\u09FF০-৯a-zA-Z0-9।,!?\'"()\s.-:;]', '', text)
    
    # Normalize punctuation
    text = re.sub(r'[।]', '। ', text)
    text = re.sub(r'([,!?:;])', r' \1 ', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short lines (likely OCR artifacts)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 5]
    
    cleaned_text = '\n'.join(cleaned_lines).strip()
    print(f"✅ Text cleaned. Length: {len(cleaned_text)} characters")
    
    return cleaned_text


def load_text_file(file_path):
    """Load text file with error handling"""
    print(f"📄 Loading text from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        
        print(f"✅ Text loaded successfully. Size: {len(raw_text)} characters")
        return raw_text
    
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        raise


# Main execution
print("🚀 Creating Knowledge Base from PDF Text")
print("=" * 50)

# 1. Load text from file
raw_text = load_text_file("E:\\Assesment_Code\\pdf_text.txt")

# 2. Clean text
cleaned_text = clean_bangla_text(raw_text)

# Free up memory
del raw_text
gc.collect()

# 3. Save cleaned text
print("💾 Saving cleaned text...")
with open("new_bangla_cleaned_ocr.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)
print("✅ Cleaned text saved to: new_bangla_cleaned_ocr.txt")

# 4. Create text chunks with optimized settings for large files
print("📝 Creating text chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,  # Slightly larger chunks for efficiency
    chunk_overlap=100,  # Reduced overlap to save memory
    separators=["\n\n", "।", "\n", " ", ""],
    length_function=len
)

chunks = text_splitter.split_text(cleaned_text)
print(f"✅ Created {len(chunks)} text chunks")

# Free up memory
del cleaned_text
gc.collect()

# Preview first chunk
if chunks:
    print(f"📖 Sample chunk:")
    print("-" * 40)
    print(chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0])
    print("-" * 40)

# 5. Load embeddings with optimization
print("🔄 Loading embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-mistral-7b-instruct",
       
    )
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    raise

# 6. Create vector store in batches for large datasets
print("🔄 Creating vector store...")
try:
    # Process chunks in smaller batches to avoid memory issues
    batch_size = 100  # Process 100 chunks at a time
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    print(f"📊 Processing {len(chunks)} chunks in {total_batches} batches...")
    
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"   Batch {batch_num}/{total_batches}: Processing {len(batch_chunks)} chunks...")
        
        # Create documents for this batch
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": "pdf_text.txt",
                    "chunk_id": i + j,
                    "batch": batch_num
                }
            )
            for j, chunk in enumerate(batch_chunks)
        ]
        
        if vectorstore is None:
            # Create initial vectorstore
            vectorstore = FAISS.from_documents(docs, embedding_model)
        else:
            # Add to existing vectorstore
            batch_store = FAISS.from_documents(docs, embedding_model)
            vectorstore.merge_from(batch_store)
        
        # Clean up memory after each batch
        del docs
        gc.collect()
        
        print(f"   ✅ Batch {batch_num} completed")
    
    print("✅ Vector store created successfully!")
    
except Exception as e:
    print(f"❌ Error creating vector store: {e}")
    raise

# 7. Save vector store
print("💾 Saving vector store...")
try:
    vectorstore.save_local("knowledge_base_bangla")
    print("✅ Vector store saved to: knowledge_base_bangla")
except Exception as e:
    print(f"❌ Error saving vector store: {e}")
    raise

# 8. Test the vector store
print("🔍 Testing vector store...")
test_queries = [
    "অপরিচিতা গল্পের লেখক কে?",
    "অপরিচিতা গল্পে অনুপম কোথায় চাকরি করতেন?",
    "What is the name of the female protagonist in অপরিচিতা?"
]

try:
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"   Query: '{query}' -> Found {len(results)} results")
        if results:
            preview = results[0].page_content[:100].replace('\n', ' ')
            print(f"     Preview: {preview}...")
    
    print("✅ Vector store test completed successfully!")
    
except Exception as e:
    print(f"❌ Error testing vector store: {e}")

print("\n🎉 Knowledge base creation completed!")
print(f"📁 Files created:")
print(f"   - new_bangla_cleaned_ocr.txt (cleaned text)")
print(f"   - new_faiss_index_bangla_test/ (vector database)")
print(f"📊 Statistics:")
print(f"   - Total chunks: {len(chunks)}")
print(f"   - Processing completed in batches")






