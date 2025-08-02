import numpy as np

def retrieve_similar_chunks(question_vector, faiss_index, chunks, k=3):
    try:
        _, indices = faiss_index.search(np.array(question_vector), k)
        return [chunks[i] for i in indices[0]]
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return []