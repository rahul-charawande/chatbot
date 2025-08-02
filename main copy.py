from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from text_chunker import chunk_text
from embedder import embed_chunks, build_faiss_index, embed_question
from retriever import retrieve_similar_chunks
from pdf_reader import extract_text_from_pdf
from generator import generate_answer
from typing import List
import os

# List of PDF paths instead of a single one
PDF_PATHS = [
    "d/Planning standards for new DP.pdf",
    "d/The Mahrashtra Development Plans Rules.pdf",
    "d/town.pdf",
    "d/UDCPR Updated 2025 (1).pdf"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store metadata about which chunk came from which document
chunks_metadata = []

print("🟢 Loading and indexing PDFs...")
all_chunks = []
for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        print(f"⚠️ Warning: PDF not found at {pdf_path}")
        continue
        
    print(f"Processing {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    all_chunks.extend(chunks)
    
    # Store metadata (document source for each chunk)
    for chunk in chunks:
        chunks_metadata.append({
            "source": os.path.basename(pdf_path),
            "content": chunk
        })

if not all_chunks:
    raise Exception("No PDFs were loaded. Please check your PDF paths.")

embeddings = embed_chunks(all_chunks)
faiss_index = build_faiss_index(embeddings)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question):
    print(f"📩 Received Question: {q.question}")
    q_vector = embed_question(q.question)
    relevant_indices = retrieve_similar_chunks(q_vector, faiss_index, k=3)  # Get top 3 chunks
    
    relevant_chunks = [all_chunks[idx] for idx in relevant_indices]
    relevant_sources = [chunks_metadata[idx]["source"] for idx in relevant_indices]
    
    context = "\n".join(relevant_chunks)
    answer = generate_answer(context, q.question)
    
    print(f"✅ Answer: {answer}")
    print(f"📚 Sources: {', '.join(set(relevant_sources))}")  # Show which documents were used
    
    return {
        "answer": answer,
        "sources": list(set(relevant_sources))  # Return unique source names
    }

@app.get("/")
def health():
    return {"status": "OK"}