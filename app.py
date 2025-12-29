import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

# LangChain & AI Imports
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Local imports from your src folder
from src.helper import download_hugging_face_embeddings, format_docs
from src.prompt import *

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 1. Initialize Embeddings (Important: Added parentheses to invoke the function)
embeddings = download_hugging_face_embeddings()

# 2. Connect to Pinecone Index
index_name = "medicalbot-end-to-end"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# 3. Setup Retriever (Retrieves top 3 similar documents)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# 4. Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 5. Build RAG Chain (Ensure "input" matches your prompt template variable)
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    """Renders the main chat interface."""
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handles user queries and returns AI responses."""
    msg = request.form["msg"]
    
    # Process the query through the LangChain RAG pipeline
    response = chain.invoke(msg)
    
    return str(response)

if __name__ == '__main__':
    # Run the application on a specific port
    app.run(host="0.0.0.0", port=8080, debug=True)