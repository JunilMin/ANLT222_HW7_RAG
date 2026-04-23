from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_FILENAME = "Yosemite-Guide-51-3-508V1.pdf"
INDEX_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
DISPLAY_SOURCES = 10
OFF_TOPIC_RESPONSE = "I’m sorry, I am only authorized to talk about the provided document."


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


def load_pdf_documents(pdf_path: str = PDF_FILENAME):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_and_save_vectorstore(pdf_path: str = PDF_FILENAME, index_dir: str = INDEX_DIR):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"{pdf_path} was not found in the project folder.")

    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    vectorstore.save_local(index_dir)
    return vectorstore


def load_vectorstore(index_dir: str = INDEX_DIR):
    return FAISS.load_local(
        index_dir,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve_top_chunks(question: str, vectorstore, k: int = TOP_K):
    return vectorstore.similarity_search(question, k=k)


def answer_question(question: str, vectorstore):
    retrieved_docs = retrieve_top_chunks(question, vectorstore, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        input_variables=["context", "question", "off_topic_response"],
        template=(
            "You are a friendly assistant.\n"
            "Answer in a natural conversational style.\n\n"

            "Rules:\n"
            "1. Use ONLY the provided context\n"
            "2. Provide detailed answers\n"
            "3. If useful, include key information as a bullet list using '-'\n"
            "4. Keep it simple and helpful\n"
            "5. If not in context, say exactly:\n"
            "{off_topic_response}\n\n"

            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "question": question,
            "off_topic_response": OFF_TOPIC_RESPONSE,
        }
    )
    return response.content.strip(), retrieved_docs