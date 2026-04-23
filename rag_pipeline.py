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
DISPLAY_SOURCES = 3
OFF_TOPIC_RESPONSE = "I’m sorry, I am only authorized to talk about the provided document."


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


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
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"{pdf_path} was not found in the same folder as the script.")

    documents = load_pdf_documents(str(pdf_file))
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


def format_sources(retrieved_docs, max_sources: int = DISPLAY_SOURCES):
    sources = []

    for doc in retrieved_docs[:max_sources]:
        page_num = doc.metadata.get("page")
        page_label = f"Page {page_num + 1}" if page_num is not None else "Page unknown"

        sources.append(
            {
                "page": page_label,
                "content": doc.page_content.strip()
            }
        )

    return sources


def answer_question(question: str, vectorstore):
    retrieved_docs = retrieve_top_chunks(question, vectorstore, k=TOP_K)

    if not retrieved_docs:
        return OFF_TOPIC_RESPONSE, []

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        input_variables=["context", "question", "off_topic_response"],
        template=(
            "You are a helpful assistant.\n"
            "Answer the user's question using ONLY the context provided below.\n"
            "If the answer is not contained in the context, respond with: {off_topic_response}, Do not use outside knowledge\n"
            "You may paraphrase the context, summarize it, and answer using semantically matching information.\n"
            "If the retrieved context contains the answer in meaning, answer normally.\n"
            # "If the answer cannot be found in the retrieved context, respond with exactly:\n"
            # "{off_topic_response}\n"
            "Do not use outside knowledge.\n"
            "Do not guess.\n"
            "Do not add any extra explanation when the answer is not in the context.\n\n"
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

    answer_text = response.content.strip()

    if answer_text == OFF_TOPIC_RESPONSE:
        return OFF_TOPIC_RESPONSE, []

    sources = format_sources(retrieved_docs, max_sources=DISPLAY_SOURCES)
    return answer_text, sources