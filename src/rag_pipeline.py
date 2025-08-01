# src/rag_pipeline.py
import os
from dotenv import load_dotenv

# --- LangChain components ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- 1. Load Environment Variables ---
load_dotenv()
print("API Keys loaded.")

# --- Constants ---
DATA_PATH = "data/raw"
DB_PATH = "vectorstores/db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def create_vector_db():
    """Creates and persists a vector database from the documents."""
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    if not documents:
        print("No documents found. Please run scraper.py first.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    print("Creating vector store...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    print("Vector store created successfully.")
    return db

def main():
    """Sets up the RAG chain and runs a query."""
    # If the DB doesn't exist, create it.
    if not os.path.exists(DB_PATH):
        print("Vector store not found. Creating a new one...")
        create_vector_db()

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

    print("Setting up Multi-Query Retriever...")
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )

    print("Setting up RAG chain...")
    prompt_template = """
    Answer the question based only on the following context.
    Provide a detailed answer. If the context includes numbers or tables, extract them.
    If you don't know the answer, just say that you don't know.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- THIS IS THE NEW QUESTION ---
    print("\nRunning a test query...")
    test_query = "What is the Monthly household income ceiling?"
    response = rag_chain.invoke(test_query)

    print("\n--- LLM Final Answer ---")
    print(f"Question: {test_query}")
    print(f"Answer: {response}")

if __name__ == "__main__":
    main()
