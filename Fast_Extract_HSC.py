from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import re

def extract_text_with_pypdfloader(pdf_path):
    """
    Fast and reliable text extraction using LangChain's PyPDFLoader
    """
    print("🔄 Loading PDF with PyPDFLoader...")
    
    try:
        # Initialize PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        
        # Load and extract text from all pages
        pages = loader.load()
        
        print(f"✅ Successfully loaded {len(pages)} pages")
        
        # Combine all page content
        full_text = ""
        for i, page in enumerate(pages):
            page_content = page.page_content
            if page_content.strip():
                full_text += f"\n--- Page {i+1} ---\n{page_content}\n"
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Processed {i+1}/{len(pages)} pages...")
        
        print(f"✅ Text extraction completed!")
        print(f"📊 Total characters extracted: {len(full_text)}")
        
        return full_text
        
    except Exception as e:
        print(f"❌ PyPDFLoader failed: {e}")
        return ""

def clean_extracted_text(text):
    """
    Clean extracted text while preserving Bengali and English
    """
    # Remove page markers for cleaner text
    text = re.sub(r'\n--- Page \d+ ---\n', '\n\n', text)
    
    # Preserve Bengali, English, numbers, and punctuation
    text = re.sub(r'[^\u0980-\u09FF০-৯a-zA-Z0-9।,!?\'"()\s.-:;\n]', '', text)
    
    # Normalize spacing
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    
    # Remove very short lines (likely headers/footers)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 5]
    
    return '\n'.join(cleaned_lines).strip()

def create_fast_knowledge_base():
    """
    Create knowledge base using PyPDFLoader
    """
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    
    print("🚀 Fast HSC Knowledge Base Creation with PyPDFLoader")
    print("=" * 60)
    
    # Extract text using PyPDFLoader
    print("🔄 Extracting text...")
    raw_text = extract_text_with_pypdfloader(pdf_path)
    
    if not raw_text.strip():
        print("❌ No text extracted. PDF might be image-based and requires OCR.")
        print("💡 Try using the OCR-based approach instead.")
        return None
    
    # Clean text
    print("🔄 Cleaning extracted text...")
    cleaned_text = clean_extracted_text(raw_text)
    
    # Save extracted text
    output_file = "hsc_bangla_pypdf_extracted.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    print(f"✅ Cleaned text saved to {output_file}")
    print(f"📊 Original characters: {len(raw_text)}")
    print(f"📊 Cleaned characters: {len(cleaned_text)}")
    
    # Show sample of cleaned text
    print("\n📖 Sample cleaned text:")
    print("-" * 40)
    sample_text = cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text
    print(sample_text + "..." if len(cleaned_text) > 500 else sample_text)
    print("-" * 40)
    
    # Create chunks
    print("\n🔄 Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Larger chunks for better context
        chunk_overlap=120,  # More overlap for continuity
        separators=["\n\n", "।", "\n", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_text(cleaned_text)
    print(f"📝 Created {len(chunks)} chunks")
    
    if chunks:
        print(f"📖 Sample chunk:")
        print(f"   Length: {len(chunks[0])} characters")
        print(f"   Preview: {chunks[0][:150]}...")
    
    # Create embeddings and vector store
    print("\n🔄 Creating embeddings and vector store...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    docs = [Document(
        page_content=chunk, 
        metadata={
            "source": "HSC26-Bangla1st-Paper-PyPDFLoader", 
            "chunk_id": i,
            "extraction_method": "PyPDFLoader"
        }
    ) for i, chunk in enumerate(chunks)]
    
    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    # Save vector store
    index_name = "faiss_index_hsc_pypdf"
    vectorstore.save_local(index_name)
    print(f"✅ Vector store saved as '{index_name}'")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "রবীন্দ্রনাথ",
        "বাংলা সাহিত্য", 
        "কবিতা",
        "ব্যাকরণ"
    ]
    
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"Query: '{query}' -> {len(results)} results")
        if results:
            print(f"   Best match: {results[0].page_content[:100]}...")
    
    print(f"\n🎉 HSC Knowledge Base created successfully!")
    print(f"📁 Files created:")
    print(f"   - {output_file} (extracted text)")
    print(f"   - {index_name}/ (vector store)")
    print(f"\n💡 Next steps:")
    print(f"   1. Update Rag_Pipline.py to use '{index_name}'")
    print(f"   2. Test with your evaluation questions")
    print(f"   3. Create HSC-specific evaluation dataset")
    
    return index_name

if __name__ == "__main__":
    create_fast_knowledge_base()
