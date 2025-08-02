import numpy as np

def retrieve_similar_chunks(question_embedding, index, k=3):
    """Retrieve top k most similar chunks from the index"""
    # question_embedding is 2D (1,384), convert to 1D (384,)
    question_embedding = question_embedding.reshape(-1)
    
    # Search the index
    distances, indices = index.search(np.array([question_embedding]), k)
    
    return indices[0]  # Return the top k indices