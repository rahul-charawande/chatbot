import fitz

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""