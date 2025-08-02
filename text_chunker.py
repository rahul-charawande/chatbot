def chunk_text(text: str, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks
