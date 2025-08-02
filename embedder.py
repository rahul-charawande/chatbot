from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return model.encode(chunks, normalize_embeddings=True)

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(384)  # Use cosine similarity
    index.add(np.array(embeddings))
    return index

def embed_question(question):
    return model.encode([question], normalize_embeddings=True)
