import streamlit as st  # ✅ Add this
from pdf_reader import extract_text_from_pdf
from text_chunker import chunk_text
from embedder import embed_chunks, build_faiss_index, embed_question
from retriever import retrieve_similar_chunks
from generator import generate_answer

@st.cache_resource
def load_index(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    return chunks, index

# ---------- UI ------------
st.set_page_config(page_title="Offline PDF QA", layout="centered")

st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        .block-container {
            padding-top: 2rem;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📘 Offline PDF QA System")
st.caption("🔐 Secure | Offline | Government-use")

chunks, faiss_index = load_index("D:\Siddhesh Somvanshi Repository\Siddhesh Professional Work\Siddhesh_all_Projects\Pdf_ChatBot\offline_pdf_qa\document\document.pdf")

with st.form("qa_form"):    
    question = st.text_input("🔍 Enter your question based on the official document:")
    submitted = st.form_submit_button("Ask Question")

if submitted and question:
    q_vec = embed_question(question)
    top_chunks = retrieve_similar_chunks(q_vec, faiss_index, chunks)
    context = "\n".join(top_chunks)

    with st.spinner("Generating secure response..."):
        answer = generate_answer(context, question)

    st.subheader("✅ Answer")
    st.write(answer)

st.markdown("---")
st.markdown("📄 **Based on static PDF:** `document.pdf`  \n⚙️ **Powered locally by Ollama + Mistral**")
