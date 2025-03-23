import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import InferenceClient
import pandas as pd
import os
from agent_setup import run_agent 

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model="mistralai/Mistral-7B-Instruct-v0.1",
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="gaipl-the-ai-vengers/code/src/chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to extract text from Excel file
def extract_text_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)  # Read Excel file
    text_data = df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()  # Convert each row to a string
    return text_data

# Function to add documents to ChromaDB
def add_document_to_db(doc_text_list, doc_id):
    embeddings = embedding_model.encode(doc_text_list).tolist()
    for i, sentence in enumerate(doc_text_list):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            embeddings=[embeddings[i]],
            metadatas=[{"text": sentence}]
        )

# Function to retrieve relevant documents
def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_texts = [doc["text"] for doc in results["metadatas"][0]]
    return " ".join(retrieved_texts)

# Function to generate response using Hugging Face LLM
def generate_response(prompt, context):
    messages = [
        {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."},
        {"role": "user", "content": f"Context: {context}\n\nUser Query: {prompt}"}
    ]
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    
    return response.choices[0].message["content"] if response.choices else "Error: No valid response from Hugging Face"
 
