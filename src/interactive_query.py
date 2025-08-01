# src/interactive_query.py
import os
from dotenv import load_dotenv

# --- LangChain components ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()
print("API Keys loaded.")

# --- Constants ---
DB_PATH = "vectorstores/db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_TO_USE = "llama3-8b-8192" # We'll use Llama 3 for our interactive tests

def main():
    """
    Sets up an interactive RAG chain where you can ask questions in the terminal.
    """
    # --- Load the Vector Database ---
    if not os.path.exists(DB_PATH):
        print("Vector store not found. Please run the main evaluation script once to create it.")
        return

    print("Loading knowledge base...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever()
    print("Knowledge base loaded successfully.")

    # --- Set up the RAG Chain ---
    llm = ChatGroq(model_name=LLM_TO_USE, temperature=0)

    prompt_template = """
    Answer the question based only on the following context.
    Provide a detailed answer. If you don't know the answer, just say that you don't know.

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

    # --- Interactive Loop ---
    print(f"\n--- Interactive Query Mode with {LLM_TO_USE} ---")
    print('Type your question and press Enter. Type "exit" or "quit" to end.')

    while True:
        # Get user input from the terminal
        user_query = input("\nYour Question: ")

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting interactive mode.")
            break

        if not user_query.strip():
            continue

        # Run the query through the RAG chain
        print("\nThinking...")
        response = rag_chain.invoke(user_query)

        # Print the final answer
        print("\n--- Answer ---")
        print(response)
        print("----------------")


if __name__ == "__main__":
    main()
