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
from Memory import chat_history, get_short_term_memory, format_chat_history  # Import from Memory module

# Load environment variables
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")


# Replace OpenAI with ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key)



# 4. Better Multilingual Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if available
    encode_kwargs={'normalize_embeddings': True}
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
- শুধুমাত্র প্রদত্ত প্রাসঙ্গিক তথ্য ব্যবহার করুন
- যদি উত্তর না জানেন, বলুন "আমি জানি না"
- উত্তরটি সর্বোচ্চ ২-৩ শব্দে দিন
- কোনো অতিরিক্ত শব্দ, ব্যাখ্যা বা বাক্য গঠন করবেন না
- শুধুমাত্র মূল তথ্য/সংখ্যা/নাম দিন

উদাহরণ:
প্রশ্ন: "কত সালে প্রকাশিত হয়?" 
উত্তর: "২০০৩"
প্রশ্ন: "কতজন পণ্ডিত কাজ করেছেন?"
উত্তর: "প্রায় ১৪৫০ জন"

প্রাসঙ্গিক তথ্য:
{context}

প্রশ্ন: {question}
উত্তর:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

## ....................................................................
# 9. Configure RetrievalQA
qa_chain_new = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),  # Get more context
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=False
)

# # 10. Test with sample queries
# test_queries = [
#     " 'অপরিচিতা' গল্পের কল্যাণীর বিয়ে না করার কারণ কী ছিল?",
#     # "'অপরিচিতা' গল্পে গল্প বলায় পটু কে ?",
# ]



# with open("test_results.txt", "w", encoding="utf-8") as file:
#     for query in test_queries:
#         if not query.strip():  # Skip empty queries
#             continue

#         # Add user query to chat history
#         chat_history.append(("User", query))

#         # Retrieve short-term memory
#         short_term_memory = get_short_term_memory(chat_history)

#         # Format the memory to pass as context
#         formatted_memory = format_chat_history(short_term_memory)

#         # Use it in the query
#         result = qa_chain_new({"query": query, "context": formatted_memory})

#         # Add system response to chat history
#         chat_history.append(("System", result['result']))

#         file.write(f"\n❓ প্রশ্ন: {query}\n")
#         file.write(f"🧠 উত্তর: {result['result']}\n")
#         file.write("🔍 প্রাসঙ্গিক অংশ:\n")
#         for doc in result['source_documents'][:2]:
#             file.write(f" - {doc.page_content[:100]}...\n")