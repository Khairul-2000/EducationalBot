import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

def update_rag_for_hsc():
    """
    Update the RAG pipeline to use HSC knowledge base
    """
    load_dotenv()
    
    print("🔄 Updating RAG Pipeline for HSC Content")
    print("=" * 50)
    
    # Check available knowledge bases
    available_kbs = []
    kb_options = [
        "faiss_index_bangla_test",
        "faiss_index_hsc_fast", 
        "faiss_index_hsc_manual"
    ]
    
    for kb in kb_options:
        if os.path.exists(kb):
            available_kbs.append(kb)
            print(f"✅ Found: {kb}")
        else:
            print(f"❌ Missing: {kb}")
    
    if not available_kbs:
        print("❌ No knowledge bases found!")
        return None
    
    # Use the best available knowledge base
    if "faiss_index_hsc_manual" in available_kbs:
        selected_kb = "faiss_index_hsc_manual"
        print(f"🎯 Using manual HSC knowledge base")
    elif "faiss_index_bangla_test" in available_kbs:
        selected_kb = "faiss_index_bangla_test" 
        print(f"🎯 Using existing Bengali knowledge base")
    else:
        selected_kb = available_kbs[0]
        print(f"🎯 Using: {selected_kb}")
    
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load vector store
    try:
        vectorstore = FAISS.load_local(
            selected_kb, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded knowledge base: {selected_kb}")
    except Exception as e:
        print(f"❌ Error loading {selected_kb}: {e}")
        return None
    
    # Setup LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return None
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)
    
    # Create improved prompt for HSC content
    prompt_template = """
আপনি একজন বাংলা সাহিত্য ও ব্যাকরণের শিক্ষক। নিম্নলিখিত প্রাসঙ্গিক তথ্য ব্যবহার করে প্রশ্নের উত্তর দিন:

নিয়মাবলী:
- শুধুমাত্র প্রদত্ত প্রাসঙ্গিক তথ্য ব্যবহার করুন
- উত্তর সংক্ষিপ্ত ও সুনির্দিষ্ট রাখুন (সর্বোচ্চ ২-৩ শব্দ)
- যদি উত্তর না জানেন, বলুন "তথ্য পাওয়া যায়নি"
- অতিরিক্ত ব্যাখ্যা দেবেন না

উদাহরণ:
প্রশ্ন: "রবীন্দ্রনাথ কবে জন্মগ্রহণ করেন?"
উত্তর: "৭ মে, ১৮৬১"

প্রাসঙ্গিক তথ্য:
{context}

প্রশ্ন: {question}
উত্তর:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False
    )
    
    print("✅ QA Chain created successfully")
    
    # Test with HSC-related queries
    test_queries = [
        "রবীন্দ্রনাথ ঠাকুর কবে জন্মগ্রহণ করেন?",
        "গীতাঞ্জলি কাব্যগ্রন্থের জন্য কী পুরস্কার পেয়েছিলেন?",
        "বাংলা ভাষায় কয়টি স্বরধ্বনি আছে?",
        "রবীন্দ্রনাথের বিখ্যাত উপন্যাসের নাম কী?"
    ]
    
    print("\n🧪 Testing QA System:")
    print("-" * 30)
    
    for query in test_queries:
        try:
            result = qa_chain({"query": query})
            print(f"❓ প্রশ্ন: {query}")
            print(f"🧠 উত্তর: {result['result']}")
            print(f"📚 সোর্স: {len(result['source_documents'])} documents")
            print("-" * 30)
        except Exception as e:
            print(f"❌ Error with query '{query}': {e}")
    
    # Save updated configuration
    config_info = f"""
# Updated RAG Configuration
Knowledge Base: {selected_kb}
Last Updated: {os.path.basename(__file__)}
Status: Ready for HSC queries
"""
    
    with open("rag_config.txt", "w", encoding="utf-8") as f:
        f.write(config_info)
    
    print(f"\n🎉 RAG Pipeline updated successfully!")
    print(f"📁 Using knowledge base: {selected_kb}")
    print(f"💡 Ready for HSC Bengali literature and grammar questions")
    
    return qa_chain

if __name__ == "__main__":
    qa_chain = update_rag_for_hsc()
