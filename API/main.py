from fastapi import FastAPI
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes import router
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Educational Bot API",
    description="A Bengali RAG-based educational chatbot API",

)

# Include the router
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Educational Bot API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
