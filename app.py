import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AlloChat - Document Q&A",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'show_toast' not in st.session_state:
    st.session_state.show_toast = False

# Sidebar for file upload
with st.sidebar:
    st.title("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open("temp_doc.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process the document
        with st.spinner("Processing document..."):
            try:
                loader = PyPDFLoader("temp_doc.pdf")
                documents = loader.load()
                
                if not documents:
                    st.error("❌ No documents found in PDF. Please try another file.")
                else:
                    st.success(f"✅ Loaded {len(documents)} pages from the document.")
                    
                    # Process the documents
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    split_docs = text_splitter.split_documents(documents)
                    
                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_documents(split_docs, embeddings)
                    retriever = vector_store.as_retriever()
                    llm = ChatOpenAI(model="gpt-3.5-turbo")
                    
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever
                    )
                    
                    # Clean up the temporary file
                    os.remove("temp_doc.pdf")
                    
                    # Show toast notification
                    st.toast('🎉 Document successfully processed! You can now ask questions about it.', icon='✅')
                    
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")

# Main chat interface
st.title("🤖 AlloChat - Document Q&A")
st.markdown("Ask questions about your uploaded document!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.qa_chain is not None:
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
else:
    st.info("👈 Please upload a PDF document in the sidebar to start chatting!") 