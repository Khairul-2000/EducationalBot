from difflib import SequenceMatcher
from Rag_pipeline_for_pdf import qa_chain_new



test_set = [
    # Basic Questions (Bengali)
    {
        "query": "দীপুর চাচার সঙ্গে অপরিচিতা' গল্পের কোন চরিত্রের মিল আছে ?",
        "expected": "মামার"
    },
    {
        "query": "'অপরিচিতা' গল্পে 'কল্যাণী' বিয়েতে কোন রঙের শাড়ি পরেছে বলে অনুপম কল্পনা করে ?",
        "expected": "লাল"
    },
    {
        "query":"'অপরিচিতা' গল্পে কল্যাণীর বিয়ে না করার সিদ্ধান্তের কারণ কী ছিল ?",
        "expected": "আত্মমর্ধাদা"
    },
    {
        "query": "শ্বশুরের সামনে অনুপমের মাথা হেট করে রাখার কারণ কী ?",
        "expected": "মামার গহনা পরীক্ষার কারণে"
    },
   
]


def similarity(a, b):
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

def evaluate_rag_system(test_set, qa_chain):
    results = []
    for item in test_set:
        query = item["query"]
        expected = item["expected"]

        response = qa_chain_new({"query": query})
        answer = response["result"]
        context = " ".join([doc.page_content for doc in response["source_documents"]])

        grounded = expected in context
        sim_score = round(similarity(expected, answer), 2)

        results.append({
            "Query": query,
            "Expected": expected,
            "Generated": answer,
            "Grounded": "✅" if grounded else "❌",
            "Similarity": sim_score,
            "Context": context
        })
    
    return results


results = evaluate_rag_system(test_set, qa_chain_new)

with open("evaluation_results.txt", "w", encoding="utf-8") as file:
    for r in results:
        file.write("\n==========================\n")
        file.write(f"❓ প্রশ্ন: {r['Query']}\n")
        file.write(f"🎯 প্রত্যাশিত: {r['Expected']}\n")
        file.write(f"🧠 উত্তর: {r['Generated']}\n")
        file.write(f"📚 Grounded: {r['Grounded']}\n")
        file.write(f"📏 Similarity: {r['Similarity']}\n")
