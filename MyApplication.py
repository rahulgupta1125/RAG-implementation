import os
import streamlit as st

# Configure the Streamlit app to use a wide layout
st.set_page_config(
    page_title="📄📚 RAG Chatbot",
    layout="wide",
)

import pdfplumber
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Ensure your API key is set in the environment
# You can also set this in Streamlit secrets or via st.text_input for production use
os.environ["GOOGLE_API_KEY"] = "AIzaSyCGErw5sxApfjUHQEgXmsQpBcgs-HmHIjI"

# Function to extract text from uploaded PDF
@st.cache_data
def extract_text_from_pdf(file_path: str) -> str:
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

# Function to vectorize the text and store in FAISS
@st.cache_resource
def vectorize_text(text: str) -> FAISS:
    # Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Function to query the vectorstore and generate an answer
def query_pdf(vectorstore: FAISS, query: str) -> str:
    # Find top 3 relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    # Initialize Gemini Flash model
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = (
        f"Using the following extracted information from the uploaded PDF, answer the question.\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response

# Streamlit UI
st.title("📄📚 RAG Chatbot on Streamlit with Google Gemini")

# PDF uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract and vectorize text
    with st.spinner("Extracting and indexing PDF..."):
        raw_text = extract_text_from_pdf(tmp_path)
        st.success("Text extracted from PDF.")
        vectorstore = vectorize_text(raw_text)
        st.success("PDF content indexed.")

    # Chat interface
    user_query = st.text_input("Ask a question about the PDF content:")
    if st.button("Get Answer") and user_query:
        with st.spinner("Generating answer..."):
            answer = query_pdf(vectorstore, user_query)
        st.markdown(f"**Answer:** {answer}")

    # Display raw text toggle
    if st.checkbox("Show extracted text"):
        st.text_area("Extracted Text", raw_text, height=300)

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except Exception:
        pass

else:
    st.info("Please upload a PDF to get started.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, LangChain, FAISS, and Google Gemini Flash")
