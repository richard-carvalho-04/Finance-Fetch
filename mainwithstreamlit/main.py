import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
os.environ["GOOGLE_API_KEY"] = "AIzaSyAUXhTHiuorSLV5Jih5y-TGIKpzvzZlZ1I"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# Streamlit UI setup
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_docs"

main_placeholder = st.empty()

def process_data(urls):
    """Load, split, and embed the data, and save to FAISS."""
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    vectorstore_gemini.save_local(file_path)  # Save FAISS index
    return vectorstore_gemini

def gemini_qa_with_sources(question, vector_store, llm):
    """Perform QA with context and sources."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    full_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context:"
    response = llm.generate_content(full_prompt)

    sources = list(set([doc.metadata.get('source', 'Unknown') for doc in relevant_docs]))
    return {
        "answer": response.text,
        "sources": sources
    }

# Handle URL processing
if process_url_clicked:
    vectorstore = process_data(urls)

# Question input and response generation
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(
            file_path, embeddings, allow_dangerous_deserialization=True
        )
        llm = genai.GenerativeModel('gemini-1.5-flash')
        result = gemini_qa_with_sources(query, vector_store, llm)

        # Display results
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", [])
        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(source)
