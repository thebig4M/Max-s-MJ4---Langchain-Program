# SkiCoach Grounded LangChain Program
# AI-assisted: Structure inspired by LangChain documentation and class examples,
# adapted and customized by the student.

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
import os

# -----------------------------
# Important Variables
# -----------------------------
PDF_PATH = "skiing.pdf"
DB_DIR = "./ski_chroma_db"
MODEL_NAME = "skicoach"

# -----------------------------
# Model + Prompt
# -----------------------------
model = ChatOllama(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_template(
    """
    You are SkiCoach, a technical ski instructor.

    You are SkiCoach, a friendly and knowledgeable ski instructor.

    If the user's question is about the provided document, you MUST use ONLY the provided context.
    If the answer is not supported by the document, respond exactly with:
    "I don't have enough context to answer that."

    If the user's question is general skiing advice and does NOT require the document,
    you may answer using your skiing expertise, emphasizing safety and progressive drills.

    Never answer questions unrelated to skiing.


    Emphasize safety and progressive skill development.

    Question:
    {input}

    Context:
    {context}

    Answer:
    """
)

parser = StrOutputParser()
chain = prompt | model | parser

# -----------------------------
# PDF Handling
# -----------------------------
def load_pdf():
    loader = PyPDFLoader(PDF_PATH)
    return loader.load_and_split()

def chunk_pdf(document):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(document)
    print(f"Split {len(document)} pages into {len(chunks)} chunks")
    return chunks

# -----------------------------
# Vector Store
# -----------------------------
def build_or_load_vectors(chunks):
    embeddings = FastEmbedEmbeddings()

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("Loading existing vector database...")
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector database...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )

    return vector_store

def format_documents(docs):
    return "\n".join(doc.page_content for doc in docs)

# -----------------------------
# Ask Function
# -----------------------------
def ask(query, retriever):
    docs = retriever.invoke(query)
    context = format_documents(docs)
    result = chain.invoke({"input": query, "context": context})
    print("\nSkiCoach:", result)
    print("-" * 60)

# -----------------------------
# MAIN LOOP
# -----------------------------
print("SkiCoach is running. Type 'exit' to quit.\n")

pdf_pages = load_pdf()
document_chunks = chunk_pdf(pdf_pages)
vectors = build_or_load_vectors(document_chunks)

retriever = vectors.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 2,
        "score_threshold": 0.5
    }
)

user_input = input("What is your skiing question?\n\n")

while user_input.lower() != "exit":
    ask(user_input, retriever)
    user_input = input("What is your skiing question?\n\n")
