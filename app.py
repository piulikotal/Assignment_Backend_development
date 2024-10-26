import streamlit as st
import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("AI Chat Support")

# Function to extract text from PDFs with truncation
def extract_text_from_pdf(pdf_file, max_length=800):
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)
        text += page.get_text()
        if len(text) > max_length:
            text = text[:max_length]  # Truncate if exceeding max_length
            break
    return text

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write("PDFs uploaded successfully.")
    
    # Extract text and convert it to Document objects
    docs = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        docs.append(Document(page_content=text))
    
    # Initialize vector store with Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    
    # Initialize Hugging Face LLM with a larger token limit model
    llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-1.3B", model_kwargs={"temperature": 0.2, "max_new_tokens": 40})
   
    # Create RetrievalQA chain with map_reduce chain_type for more concise output
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever)

    # Get user question and process it through the QA chain
    question = st.text_input("Ask a question:")
    if question:
        # Use the question to retrieve the answer
        response = qa({"query": question})
        
        # Display the full response for debugging
        st.write("Full Response:", response)
        
        # Attempt to extract just the specific answer from the response
        full_text = response.get("result", "No answer found.").strip()
        
        # Use regex to find "Answer:" or "Helpful Answer:" followed by the answer text
        match = re.search(r"(?:Helpful Answer|Answer):\s*(.*)", full_text)
        answer = match.group(1).strip() if match else "No answer found."
        
        st.write("Answer:", answer)

else:
    st.write("Please upload at least one PDF file.")