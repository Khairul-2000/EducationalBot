from pdf2image import convert_from_path
import re
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")


# Replace OpenAI with ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)



# 4. Better Multilingual Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if available
    encode_kwargs={'normalize_embeddings': True}
)


# 6. Load vector store
vectorstore = FAISS.load_local(
    "faiss_index_bangla_test", 
    embedding_model, 
    allow_dangerous_deserialization=True
)

# 7. Improved Prompt Template (Bengali-specific)
prompt_template = """
আপনাকে নিম্নলিখিত প্রাসঙ্গিক তথ্য ব্যবহার করে প্রশ্নের উত্তর দিতে হবে:
- শুধুমাত্র প্রদত্ত প্রাসঙ্গিক তথ্য ব্যবহার করুন
- যদি উত্তর না জানেন, বলুন "আমি জানি না"
- উত্তরটি সংক্ষিপ্ত এবং সুনির্দিষ্ট রাখুন

প্রাসঙ্গিক তথ্য:
{context}

প্রশ্ন: {question}
নির্ভুল উত্তর:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# # 8. Better LLM for Bengali
# llm = ChatOllama(
#     model="llama3:8b",  # Better for Bengali than Mistral
#     temperature=0.2,
#     num_ctx=4096  # Larger context window
# )


## ....................................................................
# 9. Configure RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Get more context
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=False
)

# 10. Test with sample queries
test_queries = [
    "বাংলাপিডিয়ার প্রধান উদ্দেশ্য কী?",
    "How has বাংলাপিডিয়া been received since its first publication?",
    "What should readers do if they find any errors or inconsistencies in the online version of বাংলাপিডিয়া?"
]

for query in test_queries:
    result = qa_chain({"query": query})
    print(f"\n❓ প্রশ্ন: {query}")
    print(f"🧠 উত্তর: {result['result']}")
    print("🔍 প্রাসঙ্গিক অংশ:")
    for doc in result['source_documents'][:2]:
        print(f" - {doc.page_content[:100]}...")