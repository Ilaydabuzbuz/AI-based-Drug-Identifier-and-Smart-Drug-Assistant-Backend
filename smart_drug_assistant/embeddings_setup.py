import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PDF_FOLDER = "smart_drug_assistant/patient_leaflets"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_and_chunk_pdfs(pdf_folder):
    docs = []
    metadata = []

    for fname in os.listdir(pdf_folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, fname)
        print(f"Loading PDF: {fname}")
        loader = PyPDFLoader(path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                  chunk_overlap=CHUNK_OVERLAP)

        for i, page in enumerate(pages):
            chunks = splitter.split_text(page.page_content)
            for j, chunk in enumerate(chunks):
                docs.append(chunk)
                metadata.append({
                    "source": fname,
                    "page": i + 1,
                    "chunk_id": j,
                    "text_snippet": chunk[:200] 
                })

    print(f"Loaded {len(docs)} chunks from PDFs")
    return docs, metadata

def get_vector_store():
    # Get Ollama base URL from environment variable
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing Chroma DB...")
        vector_store = Chroma(persist_directory=CHROMA_DIR,
                              embedding_function=embedding)
    else:
        docs, metadata = load_and_chunk_pdfs(PDF_FOLDER)
        print("Creating new Chroma DB...")
        vector_store = Chroma.from_texts(
            texts=docs,
            embedding=embedding,
            metadatas=metadata,
            persist_directory=CHROMA_DIR
        )
        vector_store.persist()
    return vector_store

if __name__ == "__main__":
    get_vector_store()
    