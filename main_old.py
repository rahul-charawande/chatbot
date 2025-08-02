from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from text_chunker import chunk_text
from embedder import embed_chunks, build_faiss_index, embed_question
from retriever import retrieve_similar_chunks
from pdf_reader import extract_text_from_pdf
from generator import generate_answer

PDF_PATH = "document/document.pdf"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🟢 Loading and indexing PDF...")
text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(text)
embeddings = embed_chunks(chunks)
faiss_index = build_faiss_index(embeddings)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question):
    print(f"📩 Received Question: {q.question}")
    q_vector = embed_question(q.question)
    relevant_chunks = retrieve_similar_chunks(q_vector, faiss_index, chunks)
    context = "\n".join(relevant_chunks)
    answer = generate_answer(context, q.question)
    print(f"✅ Answer: {answer}")
    return {"answer": answer}

@app.get("/")
def health():
    return {"status": "OK"}