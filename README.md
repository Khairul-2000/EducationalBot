# 🤖 Educational Bot - Bengali RAG-based Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot designed for Bengali educational content. This system processes PDF documents, creates embeddings, and provides intelligent responses using OpenAI's GPT-4 with memory capabilities.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [API Documentation](#api-documentation)
- [Sample Queries & Outputs](#sample-queries--outputs)
- [Evaluation Metrics](#evaluation-metrics)
- [Used Tools & Libraries](#used-tools--libraries)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- 📚 **PDF Knowledge Base**: Process and index Bengali PDF documents
- 🧠 **RAG Pipeline**: Intelligent retrieval and generation using FAISS vector store
- 💭 **Memory System**: Short-term conversation memory for context awareness
- 🌐 **REST API**: FastAPI-based web service with comprehensive endpoints
- 📊 **Evaluation System**: Built-in evaluation metrics for performance assessment
- 🔍 **OCR Support**: PaddleOCR integration for image-based text extraction
- 🗂️ **Multilingual**: Optimized for Bengali with English support

## 🔧 Prerequisites

- Python 3.8+ (Recommended: Python 3.12)
- Git
- OpenAI API Key
- Minimum 8GB RAM (16GB recommended for large documents)
- Windows/Linux/MacOS

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Khairul-2000/EducationalBot.git
cd EducationalBot
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv myvenv
myvenv\Scripts\activate

# Linux/MacOS
python3 -m venv myvenv
source myvenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 5: Prepare Knowledge Base

1. Place your PDF files in the project directory
2. Run the knowledge base creation script:

```bash
python Create_Knowledge_base_pdf.py
```

### Step 6: Start the API Server

```bash
# Option 1: Using Python directly
cd API
python main.py

# Option 2: Using Uvicorn
uvicorn API.main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

## 📁 Project Structure

```
EducationalBot/
├── API/
│   ├── main.py                 # FastAPI application entry point
│   └── routes.py               # API route definitions
├── faiss_index_bangla_test/    # FAISS vector store
│   ├── index.faiss
│   └── index.pkl
├── Test_Text/                  # Test files and evaluation scripts
├── myvenv/                     # Virtual environment
├── Create_Knowledge_base_pdf.py # Knowledge base creation
├── Rag_pipeline_for_pdf.py    # RAG pipeline implementation
├── Memory.py                   # Chat memory management
├── evaluate_pdf_pipeline.py   # Evaluation utilities
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## ⚙️ Configuration

### OpenAI Configuration

- Model: GPT-4o
- Temperature: 0 (for consistent responses)
- Embedding Model: HuggingFace multilingual-e5-base

### Vector Store Configuration

- FAISS index for fast similarity search
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

## 🏃‍♂️ Running the Project

### 1. Start the Knowledge Base Creation

```bash
python Create_Knowledge_base_pdf.py
```

### 2. Test the RAG Pipeline

```bash
python Rag_pipeline_for_pdf.py
```

### 3. Run API Server

```bash
cd API
python main.py
```

### 4. Access API Documentation

Visit: `http://localhost:8000/docs` for interactive API documentation

## 📖 API Documentation

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Query Processing

**POST** `/query`

Process a question using the RAG pipeline.

**Request Body:**

```json
{
  "question": "'অপরিচিতা' গল্পের কল্যাণীর বিয়ে না করার কারণ কী ছিল?"
}
```

**Response:**

```json
{
  "question": "'অপরিচিতা' গল্পের কল্যাণীর বিয়ে না করার কারণ কী ছিল?",
  "answer": "\"আত্মমর্যাদা\"",
  "source_documents": [
    "। এখানে যৌতুককে প্রত্যাখ্যান করে বিয়ে না করার সিদ্ধান্তে সবিতার অটল থাকার কথা বলা হয়েছে। উদ্দীপকের এ বিষয়টি 'অপরিচিতা' গল্পের কল্যাণীর বিয়ে আলা ব্যাট না করার সিদ্ধান্তে অটল থাকার সঙ্গে সাদৃশ্যপূর্...",
    "। 'অপরিচিতা' গল্পের প্রথমার্ধে অনুপম চরিত্রে অনুরূপ সীমাবদ্ধতা লক্ষ করা যায়। সে অন্যায় জানা সত্তেও প্রতিবাদ না করে বিয়ের আসর থেকে অপমানিত হয়ে ফিরে যায়। অনুপম একটি নির্দিষ্ট বলয়ে আটকে ছিল , কিন্ত...",
    "। এমনকি বিয়ের সভা থেকে আতিথেয়তা সম্পন্ন করে কল্যাণীর পিতা সেন মেয়েকে ব্যক্তিত্বহীন ছেলের কাছে সম্প্রদানে অসন্মতি জানান। তাই বলা যায় , উদ্দীপকের পরেশ \"'অপরিচিতা' গল্পের অনুপম চরিত্রের বিপরীত। ঘ 'অপ..."
  ],
  "memory_context": "User: 'অপরিচিতা' গল্পের কল্যাণীর বিয়ে না করার কারণ কী ছিল?"
}
```

#### 2. Test Queries

**GET** `/test`

Run predefined test queries.

**Response:**

```json
{
  "test_results": [
    {
      "question": "'অপরিচিতা' গল্পের কল্যাণীর বিয়ে না করার কারণ কী ছিল?",
      "answer": "\"আত্মমর্যাদা\"",
      "source_count": 10
    },
    {
      "question": "'অপরিচিতা' গল্পে গল্প বলায় পটু কে?",
      "answer": "\"কথক\"",
      "source_count": 10
    },
    {
      "question": "বরযাত্রীদের খাওয়া শেষ হওয়ার পর শস্তুনাথবাবু কী করতে বলেছিলেন?",
      "answer": "\"খাইতে\"",
      "source_count": 10
    }
  ]
}
```

#### 3. Evaluation

**POST** `/evaluate`

Evaluate the RAG pipeline performance.

**Request Body (Default Test Set):**

```json
{
  "test_set": [
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
}
```

**Response:**

```json
{
  "total_queries": 5,
  "average_similarity": 0.58,
  "grounded_count": 3,
  "grounded_percentage": 60.0,
  "results": [
    {
      "query": "দীপুর চাচার সঙ্গে অপরিচিতা' গল্পের কোন চরিত্রের মিল আছে?",
      "expected": "মামার",
      "generated": "\"শস্তুনাথ\"",
      "grounded": "✅",
      "similarity": 0.13,
      "context": "। 'অপরিচিতা' গল্পের শীর্ষমুহূর্ত (গ্রন্থিবন্ধন) কোনটি ? ঢা বো (ক) শস্তুনাথ সেনের কন্যা সম্প্রদানে অসম্মতির ক্ষণ (খ) ট্রেনে কল্যাণীর সাক্ষাৎলাভ মুহূর্ত (গ) সেকরা কর্তৃক গহনা পরীক্ষার মুহূর্ত (ঘ) গায়েহলুদ মুহূর্ত উত্তর ক ৩। \"যে গাছে সে ফুটিয়াছে সেই গাছকে সে একেবারে অতিক্রম করিয়া উঠিয়াছে । রবীন্দ্রনাথের ছোটগল্প রচনার স্বর্ণযুগ জাবি ইউনিট ২০১৯২০ (ক) সিরাজগঞ্জের শাহজাদপুর (খ) কুষ্টিয়ার শিলাইদহ (গ) শান্তিনিকেত (ঘ) খুলনার দক্ষিণডিহি উত্তর খ 'অপরিচিতা' গল্পে বিয়েবাড়ি যাত্রাকালে নিচের কোন যন্ত্রটি..."
    },
    {
      "query": "'অপরিচিতা' গল্পে 'কল্যাণী' বিয়েতে কোন রঙের শাড়ি পরেছে বলে অনুপম কল্পনা করে?",
      "expected": "লাল",
      "generated": "লাল",
      "grounded": "✅",
      "similarity": 1.0,
      "context": "। 'অপরিচিতা' গল্পে 'কল্যাণী' বিয়েতে কোন রঙের শাড়ি পরেছে বলে অনুপম কল্পনা করে ? কু বো '২২ (ক) হলুদ (খ) বেগুনি (গ) নীল (ঘ) লাল উত্তর ঘ 'অপরিচিতা' গল্পে কল্যাণীর বিয়ে না করার সিদ্ধান্তের কারণ কী ছিল ? চবো' ২২ (ক) লোকলজ্জা (খ) পিতৃ আদেশ (গ) আত্মমর্ধাদা (ঘ) অপবাদ উত্তর গ ৯ । এঁ মুহূর্ত হলো গল্পের শীর্ষ ! ২৬। অপরিচিতা' গল্পের কল্যাণীর বিয়ে না করার কারণ কী ছিল ? চ বো '১৬ (ক) লোকসজ্জা (খ) অপবাদ (গ) পিতার আদেশ (ঘ) আত্মমর্ধাদা উত্তর ঘ ব্যাখ্যা বিয়ের আসরে বসা কন্যার গা থেকে গহনা খুলে এনে সেকরাকে দিয়ে..."
    },
    {
      "query": "'অপরিচিতা' গল্পে কল্যাণীর বিয়ে না করার সিদ্ধান্তের কারণ কী ছিল?",
      "expected": "আত্মমর্যাদা",
      "generated": "\"যৌতুক পরীক্ষা\"",
      "grounded": "❌",
      "similarity": 0.23,
      "context": "। 'অপরিচিতা' গল্পের প্রথমার্ধে অনুপম চরিত্রে অনুরূপ সীমাবদ্ধতা লক্ষ করা যায়। সে অন্যায় জানা সত্তেও প্রতিবাদ না করে বিয়ের আসর থেকে অপমানিত হয়ে ফিরে যায়। অনুপম একটি নির্দিষ্ট বলয়ে আটকে ছিল , কিন্তু গল্পের শেষে অনুপম তার মা ও মামার তৈরি দেয়াল ভাঙতে সক্ষম হয়েছে। অপরিচিতা' গল্পে অবশেষে অনুপম তার মামা এবং মামার পরামর্শ ত্যাগ করে পূর্বের কৃতকর্মের জন্য ক্ষমাপ্রার্থনা করে কল্যাণী ও তার পিতার কাছে । এখানে যৌতুককে প্রত্যাখ্যান করে বিয়ে না করার সিদ্ধান্তে সবিতার অটল থাকার কথা বলা হয়েছে। উদ্দীপকের..."
    },
    {
      "query": "শ্বশুরের সামনে অনুপমের মাথা হেট করে রাখার কারণ কী?",
      "expected": "মামার গহনা পরীক্ষার কারণে",
      "generated": "মামার গহনা পরীক্ষার কারণে",
      "grounded": "✅",
      "similarity": 1.0,
      "context": "। শ্বশুরের সামনে অনুপমের মাথা হেট করে রাখার কারণ কী ? কু , বো '১৯ (ক) শ্বশুড়ের ব্যবহারে (খ) লজ্জায় (গ) বিয়ের আয়োজন দেখে (ঘ) মামার গহনা পরীক্ষার কারণে উত্তর ঘ ১৮। রবীন্দ্রনাথ ঠাকুর কত খ্রিষ্টাব্দে জন্মগ্রহণ করেন ? চ যো ১৯ (ক) ১৮৩৮ (খ) ১৮৪১ (গ) ১৮৬১ (ঘ) ১৮৯৯ উত্তর গ রবীন্দ্রনাথ ঠাকুর কত খ্রিষ্টাব্দে মৃত্যুবরণ করেন ? সি বো '১৯ (ক) ১৮৯১ (খ) ১৮৯৪ (গ) ১৯৪১ (ঘ) ১৯৪৬ উত্তর গ ২০ ক ? ৯০ ৯২ ঘ ৯৩ ৯৪ ক ৯৫ খ খ ৯৭ ক ৯৮ ঘ ৯৯ ঘ ১০০ ) সৃজনশীল প্রশ্ন ১ কন্যার বাপ সবুর করিতে পারিতেন , কিন্তু বরের বাপ সবুর করিতে..."
    },
    {
      "query": "বরযাত্রীদের খাওয়া শেষ হওয়ার পর শস্তুনাথবাবু কী করতে বলেছিলেন?",
      "expected": "খেতে",
      "generated": "\"খাইতে\"",
      "grounded": "❌",
      "similarity": 0.55,
      "context": "। লগ্ন\" বলিলেন , সেজন্য কিছু ভাবিবেন না এখন উঠুন। লোকটি নেহাত ভালোমানুষধরনের , কিন্তু ভিতরে বেশ একটু জোর আছে বলিয়া বোধ হইল। মামাকে উঠিতে হইল। বরযাত্রীদেরও আহার হইয়া গেল। আয়োজনের আড়ম্বর ছিল কিন্তু রান্না ভালো এবং সমস্ত বেশ পরিষ্কার পরিচ্ছন্ন বলিয়া সকলেরই তৃপ্তি হইল। বরযাত্রীদের খাওয়া শেষ হইলে শস্তুনাথবাবু আমাকে খাইতে বলিলেন। মামা বলিলেন , সে কি বিবাহের পূর্বে বর খাইবে কেমন করিয়া । সেকরা জানায় এ দুলে সোনার পরিমাণ অনেক কম আছে। এঁ কানের দুল অনুপমের মামা মেয়েকে আশীর্বাদ করার সময় দিয়েছিলেন।..."
    }
  ]
}
```

### Postman Collection

Access the complete API collection: [Postman Documentation](https://documenter.getpostman.com/view/29264276/2sB34oDHg2)

## ️ Used Tools & Libraries

### Core AI/ML Libraries:

- **LangChain**: RAG pipeline framework
- **OpenAI**: GPT-4o for text generation
- **FAISS**: Vector similarity search
- **HuggingFace Transformers**: Multilingual embeddings
- **Sentence Transformers**: Text embeddings

### Document Processing:

- **PyPDF**: PDF text extraction
- **PyMuPDF**: Advanced PDF processing
- **PaddleOCR**: Optical Character Recognition
- **pdf2image**: PDF to image conversion
- **Pillow**: Image processing

### Web Framework:

- **FastAPI**: Modern web API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Utilities:

- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **python-dotenv**: Environment variable management
- **BeautifulSoup4**: Web scraping
- **tqdm**: Progress bars

### Development Tools:

- **OpenCV**: Computer vision
- **Requests**: HTTP client
- **Tesseract**: OCR engine

## 🔧 Troubleshooting

### Common Issues:

#### 1. ModuleNotFoundError

```bash
# Ensure virtual environment is activated
myvenv\Scripts\activate  # Windows
source myvenv/bin/activate  # Linux/MacOS

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. OpenAI API Key Error

```bash
# Check .env file exists and contains:
OPENAI_API_KEY=your_actual_api_key_here
```

#### 3. FAISS Index Not Found

```bash
# Recreate the knowledge base
python Create_Knowledge_base_pdf.py
```

#### 4. Memory Issues with Large PDFs

- Increase system RAM allocation
- Process PDFs in smaller chunks
- Use pagination for large knowledge bases

#### 5. Port Already in Use

```bash
# Use different port
uvicorn API.main:app --port 8001
```

### Performance Optimization:

1. **Use GPU**: Install `faiss-gpu` for faster similarity search
2. **Batch Processing**: Process multiple documents in batches
3. **Caching**: Implement Redis for response caching
4. **Database**: Use PostgreSQL with pgvector for large-scale deployment

**Created by:** Khairul Islam  
**Repository:** [EducationalBot](https://github.com/Khairul-2000/EducationalBot)  
**Last Updated:** July 2025
