from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher
import os
import sys

# Add parent directory to path to import local modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Change working directory to parent directory to find the FAISS index
os.chdir(parent_dir)

from Rag_pipeline_for_pdf import qa_chain_new
from Memory import chat_history, get_short_term_memory, format_chat_history

# Create router
router = APIRouter()

# Request and Response models
class QueryRequest(BaseModel):
    question: str
    use_memory: Optional[bool] = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[str]
    memory_context: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    chat_history: List[tuple]
    short_term_memory: List[tuple]

class EvaluationRequest(BaseModel):
    test_set: Optional[List[Dict[str, str]]] = None

class EvaluationResult(BaseModel):
    query: str
    expected: str
    generated: str
    grounded: str
    similarity: float
    context: str

class EvaluationResponse(BaseModel):
    total_queries: int
    average_similarity: float
    grounded_count: int
    grounded_percentage: float
    results: List[EvaluationResult]

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a question using the RAG pipeline
    """
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Add user query to chat history
        chat_history.append(("User", question))
        
        memory_context = None
        if request.use_memory:
            # Retrieve short-term memory
            short_term_memory = get_short_term_memory(chat_history)
            memory_context = format_chat_history(short_term_memory)
        
        # Process the query
        result = qa_chain_new({"query": question})
        answer = result['result']
        
        # Add system response to chat history
        chat_history.append(("System", answer))
        
        # Extract source documents
        source_docs = [doc.page_content[:200] + "..." for doc in result['source_documents'][:3]]
        
        return QueryResponse(
            question=question,
            answer=answer,
            source_documents=source_docs,
            memory_context=memory_context
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")



@router.get("/test")
async def test_queries():
    """
    Run test queries and return results
    """
    test_questions = [
        "'অপরিচিতা' গল্পের কল্যাণীর বিয়ে না করার কারণ কী ছিল?",
        "'অপরিচিতা' গল্পে গল্প বলায় পটু কে?",
        "বরযাত্রীদের খাওয়া শেষ হওয়ার পর শস্তুনাথবাবু কী করতে বলেছিলেন?"
    ]
    
    results = []
    
    for question in test_questions:
        try:
            result = qa_chain_new({"query": question})
            results.append({
                "question": question,
                "answer": result['result'],
                "source_count": len(result['source_documents'])
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": f"Error: {str(e)}",
                "source_count": 0
            })
    
    return {"test_results": results}

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag_pipeline(request: EvaluationRequest):
    """
    Evaluate the RAG pipeline with a test set
    """
    try:
        # Default test set if none provided
        default_test_set = [
            {
                "query": "দীপুর চাচার সঙ্গে অপরিচিতা' গল্পের কোন চরিত্রের মিল আছে?",
                "expected": "মামার"
            },
            {
                "query": "'অপরিচিতা' গল্পে 'কল্যাণী' বিয়েতে কোন রঙের শাড়ি পরেছে বলে অনুপম কল্পনা করে?",
                "expected": "লাল"
            },
            {
                "query": "'অপরিচিতা' গল্পে কল্যাণীর বিয়ে না করার সিদ্ধান্তের কারণ কী ছিল?",
                "expected": "আত্মমর্যাদা"
            },
            {
                "query": "শ্বশুরের সামনে অনুপমের মাথা হেট করে রাখার কারণ কী?",
                "expected": "মামার গহনা পরীক্ষার কারণে"
            },
            {
                "query": "বরযাত্রীদের খাওয়া শেষ হওয়ার পর শস্তুনাথবাবু কী করতে বলেছিলেন?",
                "expected": "খেতে"
            }
        ]
        
        test_set = request.test_set if request.test_set else default_test_set
        
        results = []
        total_similarity = 0
        grounded_count = 0
        
        for item in test_set:
            query = item["query"]
            expected = item["expected"]
            
            # Process the query
            response = qa_chain_new({"query": query})
            answer = response["result"]
            context = " ".join([doc.page_content for doc in response["source_documents"]])
            
            # Check if answer is grounded in context
            grounded = expected.lower() in context.lower()
            if grounded:
                grounded_count += 1
            
            # Calculate similarity
            sim_score = round(similarity(expected, answer), 2)
            total_similarity += sim_score
            
            results.append(EvaluationResult(
                query=query,
                expected=expected,
                generated=answer,
                grounded="✅" if grounded else "❌",
                similarity=sim_score,
                context=context[:500] + "..." if len(context) > 500 else context
            ))
        
        # Calculate metrics
        average_similarity = round(total_similarity / len(test_set), 2)
        grounded_percentage = round((grounded_count / len(test_set)) * 100, 1)
        
        return EvaluationResponse(
            total_queries=len(test_set),
            average_similarity=average_similarity,
            grounded_count=grounded_count,
            grounded_percentage=grounded_percentage,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")




