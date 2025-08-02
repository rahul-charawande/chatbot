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

# Directory containing PDFs
PDF_DIR = "d/"

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

# Get all PDF files in the directory
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]

if not pdf_files:
    raise Exception(f"No PDF files found in directory {PDF_DIR}")

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    print(f"Processing {pdf_path}...")
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        
        # Store metadata (document source for each chunk)
        for chunk in chunks:
            chunks_metadata.append({
                "source": pdf_file,  # Just the filename without path
                "content": chunk
            })
    except Exception as e:
        print(f"⚠️ Error processing {pdf_file}: {str(e)}")
        continue

if not all_chunks:
    raise Exception("No PDFs were successfully loaded. Please check your PDF files.")

embeddings = embed_chunks(all_chunks)
faiss_index = build_faiss_index(embeddings)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question):
    print(f"📩 Received Question: {q.question}")
    q_vector = embed_question(q.question)
    relevant_indices = retrieve_similar_chunks(q_vector, faiss_index, k=3)  # Get top 3 chunks
    
    # Get relevant chunks and their metadata
    relevant_chunks = [all_chunks[idx] for idx in relevant_indices]
    relevant_sources = [chunks_metadata[idx]["source"] for idx in relevant_indices]
    relevant_contents = [chunks_metadata[idx]["content"] for idx in relevant_indices]
    
    # Create context for generation
    context = "\n".join(relevant_chunks)
    answer = generate_answer(context, q.question)
    
    print(f"✅ Answer: {answer}")
    print(f"📚 Sources: {', '.join(set(relevant_sources))}")
    
    # Prepare the supporting content (small paragraphs)
    supporting_content = []
    for content, source in zip(relevant_contents, relevant_sources):
        # Take first 2 sentences or 200 characters (whichever comes first)
        sentences = content.split('. ')
        short_content = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else content
        short_content = short_content[:200]  # Ensure it's not too long
        
        supporting_content.append({
            "text": short_content,
            "source": source
        })
    
    return {
        "answer": answer,
        "sources": list(set(relevant_sources)),
        "supporting_content": supporting_content
    }

@app.get("/")
def health():
    return {"status": "OK"}