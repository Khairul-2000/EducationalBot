from pdf2image import convert_from_path
import re
import os
from langchain.embeddings import HuggingFaceEmbeddings
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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key)



# 4. Better Multilingual Embeddings
# Replace OpenAI embeddings with HuggingFace e5-mistral-7b-instruct embeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize HuggingFace Embeddings with e5-mistral-7b-instruct
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-mistral-7b-instruct",  # Powerful multilingual embedding model
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
)


# 6. Load vector store
vectorstore = FAISS.load_local(
    "new_faiss_index_bangla_test", 
    embedding_model, 
    allow_dangerous_deserialization=True
)

# 7. Improved Prompt Template (Bengali-specific)
prompt_template = """
আপনাকে নিম্নলিখিত প্রাসঙ্গিক তথ্য ব্যবহার করে প্রশ্নের উত্তর দিতে হবে:

গুরুত্বপূর্ণ নির্দেশনা:
1. প্রথমে যাচাই করুন যে প্রদত্ত তথ্য প্রশ্নের সাথে সরাসরি সম্পর্কিত কি না
2. যদি প্রদত্ত তথ্যে প্রশ্নের উত্তর না থাকে, তাহলে "এই তথ্য এই প্রশ্নের সাথে সম্পর্কিত নয়" বলুন
3. শুধুমাত্র প্রদত্ত তথ্য থেকে সরাসরি পাওয়া উত্তর দিন
4. কোনো অনুমান, ব্যাখ্যা বা অতিরিক্ত তথ্য যোগ করবেন না
5. উত্তর সংক্ষিপ্ত রাখুন (২-৩ শব্দ বা একটি বাক্য)

উদাহরণ:
প্রশ্ন: "রবীন্দ্রনাথ কত সালে নোবেল পেয়েছেন?" 
যদি তথ্যে আছে: "১৯১৩"
যদি তথ্যে নেই: "এই তথ্য এই প্রশ্নের সাথে সম্পর্কিত নয়"

প্রাসঙ্গিক তথ্য:
{context}

প্রশ্ন: {question}
উত্তর:
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
qa_chain_new = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "score_threshold": 0.5  # Specify the score threshold as a float
        },
        search_type="similarity_score_threshold"
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=False
)

# 10. Test with sample queries
test_queries = [
    "অপরিচিতা গল্পের লেখক কে?",
    "অপরিচিতা গল্পে অনুপম কোথায় চাকরি করতেন?",
    "What is the name of the female protagonist in অপরিচিতা?"
]

for query in test_queries:
    result = qa_chain_new({"query": query})
    print(f"\n❓ প্রশ্ন: {query}")
    print(f"🧠 উত্তর: {result['result']}")
    print("🔍 প্রাসঙ্গিক অংশ:")
    for doc in result['source_documents'][:2]:
        print(f" - {doc.page_content[:100]}...")