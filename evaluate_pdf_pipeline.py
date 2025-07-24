from difflib import SequenceMatcher
from Rag_pipeline_for_pdf import qa_chain_new



test_set = [
    # Basic Questions (Bengali)
    {
        "query": "বাংলাপিডিয়া কী?",
        "expected": "বাংলাদেশের জাতীয় জ্ঞানকোষ"
    },
    {
        "query": "বাংলাপিডিয়াতে কতজন পণ্ডিত কাজ করেছেন?",
        "expected": "প্রায় ১৪৫০ জন"
    },
    {
        "query":"How many scholars worked on the banglapedia project?",
        "expected": "প্রায় ১৪৫০ জন"
    },

    # Mixed Language Questions
    {
        "query": "What is the publication year of the first edition of বাংলাপিডিয়া?",
        "expected": "২০০৩"
    },
    {
        "query": "How many scholars worked on বাংলাপিডিয়া project?",
        "expected": "প্রায় ১৪৫০ জন"
    },
    {
        "query": "বাংলাপিডিয়ার current edition কবে published হয়েছে?",
        "expected": "২০১২"
    },
    
    # Complex Questions
    {
        "query": "বাংলাপিডিয়ার first ও second edition এর মধ্যে কত বছরের gap?",
        "expected": "৯ বছর"
    },
    {
        "query": "What should readers do if they find errors in the online version of বাংলাপিডিয়া?",
        "expected": "আমাদেরকে অবহিত করুন"
    },
    {
        "query": "বাংলাপিডিয়া কোন time period cover করে?",
        "expected": "প্রাচীনতমকাল থেকে বর্তমান পর্যন্ত"
    },
    
    # Professional/Academic Questions
    {
        "query": "Who are the primary target audience of বাংলাপিডিয়া according to the text?",
        "expected": "শিক্ষক, শিক্ষার্থী, গবেষক, পেশাজীবী এবং সাধারণ পাঠক"
    },
    {
        "query": "বাংলাপিডিয়ার reception কেমন ছিল after first publication?",
        "expected": "ব্যাপক উৎসাহব্যঞ্জক সাড়া পেয়েছে"
    },
    
    # Analytical Questions
    {
        "query": "বাংলাপিডিয়ার second edition (২০১২) এর relationship with first edition (২০০৩) কী?",
        "expected": "পরিবর্ধিত ও হালনাগাদ রূপ"
    },
    {
        "query": "What makes বাংলাপিডিয়া an essential companion for different professional groups?",
        "expected": "বাংলাদেশ সংক্রান্ত সকল বিষয়ে সুস্পষ্ট জ্ঞান"
    },
    
    # Error Detection Questions
    {
        "query": "অনলাইন বাংলাপিডিয়ায় ভুল পেলে readers দের কী করা উচিত?",
        "expected": "আমাদেরকে অবহিত করুন"
    },
    {
        "query": "How has বাংলাপিডিয়া been recognized internationally?",
        "expected": "আন্তর্জাতিকভাবে প্রশংসিত"
    },
    
    # Comprehensive Questions
    {
        "query": "বাংলাপিডিয়ার scope of coverage কী এবং এর impact কী হয়েছে?",
        "expected": "বাংলাদেশের সকল গুরুত্বপূর্ণ বিষয় এবং অপরিহার্য সহচর"
    }
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


results = evaluate_rag_system(test_set, qa_chain)

for r in results:
    print("\n==========================")
    print("❓ প্রশ্ন:", r["Query"])
    print("🎯 প্রত্যাশিত:", r["Expected"])
    print("🧠 উত্তর:", r["Generated"])
    print("📚 Grounded:", r["Grounded"])
    print("📏 Similarity:", r["Similarity"])
