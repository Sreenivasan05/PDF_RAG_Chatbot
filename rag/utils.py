
import hashlib

def get_pdf_signature(pdf_docs):
    hasher = hashlib.md5()
    for pdf in pdf_docs:
        hasher.update(pdf.getvalue())
    return hasher.hexdigest()