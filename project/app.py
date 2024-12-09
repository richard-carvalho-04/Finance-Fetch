import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Flask app setup
app = Flask(__name__)

file_path = "faiss_docs"

def process_data(urls):
    """Load, split, and embed the data, and save to FAISS."""
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    # Create embeddings and save to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
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

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")  # Requires a template called `index.html`

@app.route("/process_urls", methods=["POST"])
def process_urls():
    """Handle URL processing."""
    urls = request.json.get("urls", [])
    if not urls:
        return jsonify({"error": "No URLs provided."}), 400
    
    try:
        vectorstore = process_data(urls)
        return jsonify({"message": "Data processed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle question answering."""
    question = request.json.get("question", "")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    if os.path.exists(file_path):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local(
                file_path, embeddings, allow_dangerous_deserialization=True
            )
            llm = genai.GenerativeModel('gemini-1.5-flash')
            result = gemini_qa_with_sources(question, vector_store, llm)
            return jsonify({
                "answer": result["answer"],
                "sources": result["sources"]
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No data processed yet. Please process URLs first."}), 400

if __name__ == "__main__":
    app.run(debug=True)
