from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

import os

# Load environment variables
load_dotenv()

# Load and process documents
def load_documents():
    loader = PyPDFLoader("your_docs/your_file.pdf")  # Change this path to your actual PDF file
    documents = loader.load()
    
    if not documents:
        print("❌ No documents found in PDF. Check the file or try another one.")
        exit()

    print(f"✅ Loaded {len(documents)} documents.")
    return documents

# Set up the Q&A system
def setup_qa():
    documents = load_documents()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)

    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Or "gpt-4" if you have access

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Chat function
def chat_with_docs():
    qa = setup_qa()
    print("🗨️ Ask questions about your contract (type 'exit' to quit)\n")

    while True:
        query = input("🧾 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa.run(query)
        print(f"🤖 AlloChat: {answer}\n")

if __name__ == "__main__":
    chat_with_docs()
