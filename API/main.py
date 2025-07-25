from fastapi import FastAPI
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
