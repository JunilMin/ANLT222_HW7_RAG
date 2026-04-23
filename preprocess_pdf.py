from rag_pipeline import build_and_save_vectorstore

if __name__ == "__main__":
    print("Building FAISS index from the PDF...")
    build_and_save_vectorstore()
