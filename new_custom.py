import os
import glob
import requests
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import argparse

# Global configuration
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b"
CHROMA_DB_PATH = "./chroma_db"

def get_embedding(text):
    """
    Generate embedding for text using BGE-small-en-v1.5 model
    
    Args:
        text (str): Text to embed
        
    Returns:
        list: Embedding vector
    """
    return EMBED_MODEL.encode(text).tolist()

def ingest_documents(folder_path, collection_name="docs"):
    """
    Ingest documents from a folder into the vector database
    
    Args:
        folder_path (str): Path to folder containing documents
        collection_name (str): Name of the collection in ChromaDB
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = chromadb.Client(Settings(persist_directory=CHROMA_DB_PATH))
        collection = client.get_or_create_collection(collection_name)
        
        # Get all files in the folder
        files = glob.glob(os.path.join(folder_path, "*"))
        if not files:
            print(f"No files found in {folder_path}")
            return False
            
        for file in tqdm(files, desc="Ingesting documents"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Split text into chunks (500 characters each)
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                
                for idx, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        emb = get_embedding("passage: " + chunk)
                        collection.add(
                            documents=[chunk],
                            embeddings=[emb],
                            ids=[f"{os.path.basename(file)}_{idx}"]
                        )
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
                
        client.persist()
        print("Document ingestion completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        return False

def query_documents(question, collection_name="docs", num_results=4):
    """
    Query the document collection with a question
    
    Args:
        question (str): The question to ask
        collection_name (str): Name of the collection in ChromaDB
        num_results (int): Number of relevant chunks to retrieve
        
    Returns:
        str: Generated response from the model
    """
    try:
        client = chromadb.Client(Settings(persist_directory=CHROMA_DB_PATH))
        collection = client.get_collection(collection_name)
        
        # Generate embedding for the question
        q_emb = get_embedding("query: " + question)
        
        # Query the collection
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=num_results
        )
        
        # Combine context from relevant chunks
        context = "\n\n".join([doc for doc in results["documents"][0]])
        
        # Create prompt for the model
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # Get response from Ollama
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt}
        )
        
        return response.json()["response"]
        
    except Exception as e:
        return f"Error: {str(e)}"

def process_excel_prompts(excel_file, sheet_type, output_file=None):
    """
    Process prompts from Excel file and generate responses
    
    Args:
        excel_file (str): Path to the Excel file
        sheet_type (str): Either 'WA' or 'CA' to select the appropriate sheet
        output_file (str, optional): Path for output file. If None, auto-generates name
        
    Returns:
        str: Path to the output file, or None if failed
    """
    try:
        # Determine sheet name based on input
        if sheet_type.upper() == 'WA':
            sheet_name = 'WA_GT'
        elif sheet_type.upper() == 'CA':
            sheet_name = 'CA_GT'
        else:
            raise ValueError("sheet_type must be either 'WA' or 'CA'")
        
        # Read the Excel file
        print(f"Reading sheet '{sheet_name}' from {excel_file}...")
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Check if required columns exist
        required_columns = ['Prompt Type', 'Prompt', 'GroundTruth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Found {len(df)} prompts to process")
        
        # Initialize the response column
        df['Response'] = ''
        
        # Process each prompt
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
            prompt = row['Prompt']
            if pd.isna(prompt) or prompt.strip() == '':
                df.at[index, 'Response'] = "Error: Empty prompt"
                continue
                
            try:
                response = query_documents(prompt)
                df.at[index, 'Response'] = response
            except Exception as e:
                df.at[index, 'Response'] = f"Error: {str(e)}"
        
        # Create output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(excel_file)[0]
            output_file = f"{base_name}_{sheet_type}_with_responses.xlsx"
        
        # Save to new Excel file
        print(f"Saving results to {output_file}...")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'{sheet_name}_with_responses', index=False)
        
        print(f"Processing complete! Results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return None

def batch_query_documents(questions, collection_name="docs"):
    """
    Process multiple questions in batch
    
    Args:
        questions (list): List of questions to process
        collection_name (str): Name of the collection in ChromaDB
        
    Returns:
        list: List of responses corresponding to the questions
    """
    responses = []
    for question in tqdm(questions, desc="Processing questions"):
        response = query_documents(question, collection_name)
        responses.append(response)
    return responses

def check_system_status():
    """
    Check if the system is properly configured and running
    
    Returns:
        dict: Status information about the system
    """
    status = {
        "ollama_running": False,
        "chroma_db_exists": False,
        "embedding_model_loaded": False,
        "collection_exists": False
    }
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        status["ollama_running"] = response.status_code == 200
    except:
        pass
    
    # Check if ChromaDB exists
    status["chroma_db_exists"] = os.path.exists(CHROMA_DB_PATH)
    
    # Check if embedding model is loaded
    try:
        test_embedding = get_embedding("test")
        status["embedding_model_loaded"] = len(test_embedding) > 0
    except:
        pass
    
    # Check if collection exists
    try:
        client = chromadb.Client(Settings(persist_directory=CHROMA_DB_PATH))
        collections = client.list_collections()
        status["collection_exists"] = len(collections) > 0
    except:
        pass
    
    return status

def query_with_context(question):
    """Query the private GPT system with the given question"""
    try:
        client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        collection = client.get_collection("docs")
        q_emb = get_embedding("query: " + question)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=4
        )
        context = "\n\n".join([doc for doc in results["documents"][0]])
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # Chat with Ollama
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt}
        )
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"
    
    
def main():
    """Example integration of private GPT utilities"""
    
    # 1. Check system status first
    print("Checking system status...")
    status = check_system_status()
    print(f"System Status: {status}")
    
    # 2. Ingest documents (run this once)
    print("\n=== Document Ingestion ===")
    documents_folder = "./documents"  # Change this to your documents folder
    success = ingest_documents(documents_folder)
    if success:
        print("✅ Documents ingested successfully!")
    else:
        print("❌ Document ingestion failed!")
        return
    
    # 3. Single query example
    print("\n=== Single Query Example ===")
    question = "What is the main topic of the documents?"
    response = query_documents(question)
    print(f"Q: {question}")
    print(f"A: {response}")
    
    # 4. Batch processing example
    print("\n=== Batch Processing Example ===")
    questions = [
        "What are the key points?",
        "Summarize the main content",
        "What are the conclusions?"
    ]
    responses = batch_query_documents(questions)
    
    for i, (q, r) in enumerate(zip(questions, responses), 1):
        print(f"\n{i}. Q: {q}")
        print(f"   A: {r}")
    
    # 5. Excel processing example
    print("\n=== Excel Processing Example ===")
    excel_file = "prompts.xlsx"  # Change this to your Excel file
    sheet_type = "WA"  # or "CA"
    
    output_file = process_excel_prompts(excel_file, sheet_type)
    if output_file:
        print(f"✅ Excel processing complete! Output: {output_file}")
    else:
        print("❌ Excel processing failed!")
        
# Example usage functions
def example_usage():
    """
    Example of how to use the utility functions
    """
    print("=== Private GPT Utilities Example ===")
    
    # Check system status
    status = check_system_status()
    print("System Status:", status)
    
    # Example 1: Ingest documents
    # success = ingest_documents("./documents_folder")
    # print(f"Ingestion successful: {success}")
    
    # Example 2: Single query
    # response = query_documents("What is the main topic?")
    # print(f"Response: {response}")
    
    # Example 3: Process Excel file
    # output_file = process_excel_prompts("prompts.xlsx", "WA")
    # print(f"Output file: {output_file}")
    
    # Example 4: Batch processing
    # questions = ["Question 1", "Question 2", "Question 3"]
    # responses = batch_query_documents(questions)
    # for q, r in zip(questions, responses):
    #     print(f"Q: {q}\nA: {r}\n")
    
def custom_integration_example():
    """Example of custom integration with your own logic"""
    
    # Your custom logic here
    custom_questions = [
        "What is the company's mission?",
        "What are the main products?",
        "What is the contact information?"
    ]
    
    # Process with your own logic
    results = {}
    for question in custom_questions:
        response = query_documents(question)
        results[question] = response
        
        # Your custom processing logic here
        if "mission" in question.lower():
            print(f"Mission-related response: {response}")
        elif "product" in question.lower():
            print(f"Product-related response: {response}")
    
    return results

def api_integration_example():
    """Example of how to use in an API context"""
    
    def handle_user_query(user_question):
        """Handle a user query from an API"""
        try:
            response = query_documents(user_question)
            return {
                "success": True,
                "response": response,
                "question": user_question
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "question": user_question
            }
    
    # Example API usage
    api_response = handle_user_query("What is the main topic?")
    print(f"API Response: {api_response}")