import os
import pandas as pd
from flask import current_app
import threading
from datetime import datetime
import traceback
import glob
import requests
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Global configuration
# EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
EMBED_MODEL = SentenceTransformer("models/custom_embedding")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b"
CHROMA_DB_PATH = "./chroma_db"

# Global progress tracker
progress_tracker = {}
progress_lock = threading.Lock()

def update_progress(model_name, stage, message):
    """Update progress for a specific model evaluation."""
    with progress_lock:
        if model_name not in progress_tracker:
            progress_tracker[model_name] = {}
        progress_tracker[model_name].update({
            'stage': stage,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    print(f"📊 Progress Update - {model_name}: Stage {stage} - {message}")

def get_progress(model_name):
    """Get current progress for a model."""
    with progress_lock:
        return progress_tracker.get(model_name, {'stage': 0, 'message': 'Not started'})

def clear_progress(model_name):
    """Clear progress tracking for a specific model."""
    with progress_lock:
        if model_name in progress_tracker:
            del progress_tracker[model_name]
    print(f"🧹 Cleared progress tracking for {model_name}")

def get_embedding(text):
    """Generate embedding for text using BGE-small-en-v1.5 model"""
    return EMBED_MODEL.encode(text).tolist()

def ingest_documents(folder_path, collection_name="docs"):
    """Ingest documents from a folder into the vector database"""
    try:
        client = chromadb.Client(Settings(persist_directory=CHROMA_DB_PATH))
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
            print(f"🗑️ Deleted existing collection: {collection_name}")
        except:
            pass
        
        collection = client.get_or_create_collection(collection_name)
        
        # Get all PDF files in the folder
        files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not files:
            print(f"No PDF files found in {folder_path}")
            return False
            
        for file in tqdm(files, desc="Ingesting documents"):
            try:
                # For now, we'll assume PDF content is extracted elsewhere
                # In a real implementation, you'd use PyPDF2 or similar
                with open(file, "r", encoding="utf-8", errors='ignore') as f:
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
    """Query the document collection with a question"""
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

def calculate_similarity_score(actual, extracted):
    """Calculate similarity score between actual and extracted values"""
    if pd.isna(actual) or pd.isna(extracted):
        return 0.0
    
    actual_str = str(actual).strip().lower()
    extracted_str = str(extracted).strip().lower()
    
    # Simple similarity calculation (you can enhance this)
    if actual_str == extracted_str:
        return 100.0
    elif actual_str in extracted_str or extracted_str in actual_str:
        return 80.0
    else:
        # Use sentence transformer for semantic similarity
        actual_emb = EMBED_MODEL.encode(actual_str)
        extracted_emb = EMBED_MODEL.encode(extracted_str)
        from sklearn.metrics.pairwise import cosine_similarity
        score = cosine_similarity([actual_emb], [extracted_emb])[0][0]
        return round(score * 100, 2)

def grade_confidence(score):
    """Grade the confidence score"""
    if score >= 90:
        return '✅ Pass'
    elif score >= 70:
        return '⚠ Intermittent'
    else:
        return '❌ Fail'

def run_custom_evaluation(model_name, model_path, upload_dir):
    """Main function to run custom RAG-based evaluation"""
    print(f"🚀 Starting custom RAG evaluation for model: {model_name}")
    
    try:
        # Stage 1: Loading models and initializing
        update_progress(model_name, 1, "Loading models and initializing...")
        
        # Determine evaluation type based on folder structure
        wealth_advisory_dir = os.path.join(upload_dir, 'wealth_advisory')
        compliance_dir = os.path.join(upload_dir, 'compliance')
        
        eval_type = None
        eval_dir = None
        
        if os.path.exists(wealth_advisory_dir):
            eval_type = "wealth_advisory"
            eval_dir = wealth_advisory_dir
        elif os.path.exists(compliance_dir):
            eval_type = "compliance"
            eval_dir = compliance_dir
        else:
            raise Exception("Neither 'wealth_advisory' nor 'compliance' folder found in uploads directory")
        
        print(f"📁 Using evaluation type: {eval_type}")
        print(f"📂 Evaluation directory: {eval_dir}")
        
        # Find PDF files and Excel ground truth
        pdf_files = glob.glob(os.path.join(eval_dir, "*.pdf"))
        excel_files = glob.glob(os.path.join(eval_dir, "*.xlsx")) + glob.glob(os.path.join(eval_dir, "*.xls"))
        
        if not pdf_files:
            raise Exception(f"No PDF files found in {eval_dir}")
        if not excel_files:
            raise Exception(f"No Excel ground truth files found in {eval_dir}")
        
        pdf_file = pdf_files[0]
        excel_file = excel_files[0]
        
        print(f"📄 Using PDF: {os.path.basename(pdf_file)}")
        print(f"📊 Using Excel: {os.path.basename(excel_file)}")
        
        # Stage 2: Processing uploaded files and ingesting to ChromaDB
        update_progress(model_name, 2, "Processing files and creating vector database...")
        
        collection_name = f"{model_name}_{eval_type}"
        success = ingest_documents(eval_dir, collection_name)
        if not success:
            raise Exception("Failed to ingest documents into ChromaDB")
        
        # Stage 3: Loading ground truth and running queries
        update_progress(model_name, 3, "Loading ground truth and running queries...")
        
        # Load ground truth Excel file
        df_ground_truth = pd.read_excel(excel_file)
        if 'Prompt' not in df_ground_truth.columns or 'GroundTruth' not in df_ground_truth.columns:
            raise Exception("Excel file must contain 'Prompt' and 'GroundTruth' columns")
        
        print(f"📋 Loaded {len(df_ground_truth)} prompts from ground truth")
        
        # Stage 4: Analyzing content and generating responses
        update_progress(model_name, 4, "Analyzing content and generating responses...")
        
        results = []
        for idx, row in tqdm(df_ground_truth.iterrows(), total=len(df_ground_truth), desc="Processing prompts"):
            prompt = row['Prompt']
            ground_truth = row['GroundTruth']
            
            if pd.isna(prompt) or prompt.strip() == '':
                continue
            
            try:
                # Query the RAG system
                extracted_answer = query_documents(prompt, collection_name)
                
                # Calculate similarity score
                score = calculate_similarity_score(ground_truth, extracted_answer)
                grade = grade_confidence(score)
                
                results.append({
                    'prompt': prompt,
                    'actual': str(ground_truth),
                    'extracted': str(extracted_answer),
                    'score': float(score),
                    'grade': grade
                })
                
            except Exception as e:
                print(f"Error processing prompt: {prompt} - {str(e)}")
                results.append({
                    'prompt': prompt,
                    'actual': str(ground_truth),
                    'extracted': f"Error: {str(e)}",
                    'score': 0.0,
                    'grade': '❌ Fail'
                })
        
        # Stage 5: Finalizing results
        update_progress(model_name, 5, "Finalizing evaluation results...")
        
        if not results:
            raise Exception("No results generated from evaluation")
        
        # Calculate summary statistics
        total_tests = len(results)
        pass_count = len([r for r in results if r['grade'] == '✅ Pass'])
        intermittent_count = len([r for r in results if r['grade'] == '⚠ Intermittent'])
        fail_count = len([r for r in results if r['grade'] == '❌ Fail'])
        
        scores = [r['score'] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        overall_score = avg_score
        success_rate = (pass_count / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 Evaluation Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {pass_count} ({pass_count/total_tests*100:.1f}%)")
        print(f"   Intermittent: {intermittent_count} ({intermittent_count/total_tests*100:.1f}%)")
        print(f"   Failed: {fail_count} ({fail_count/total_tests*100:.1f}%)")
        print(f"   Average Score: {avg_score:.1f}%")
        
        # Prepare comprehensive results
        final_results = {
            "model_name": model_name,
            "evaluation_type": eval_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_processed": len(pdf_files) + len(excel_files),
            "overall_score": float(overall_score),
            "total_tests": total_tests,
            "pass_count": pass_count,
            "intermittent_count": intermittent_count,
            "fail_count": fail_count,
            "average_score": float(avg_score),
            "success_rate": float(success_rate),
            "ground_truth_comparison": results,
            "file_info": {
                "pdf_file": os.path.basename(pdf_file),
                "ground_truth_file": os.path.basename(excel_file),
                "evaluation_directory": eval_type
            },
            "summary_statistics": {
                "highest_score": float(max(scores)) if scores else 0,
                "lowest_score": float(min(scores)) if scores else 0,
                "median_score": float(sorted(scores)[len(scores)//2]) if scores else 0,
                "std_deviation": float(pd.Series(scores).std()) if scores else 0
            }
        }
        
        # Mark as completed
        update_progress(model_name, 6, "Evaluation completed successfully!")
        
        print("🎉 Custom RAG evaluation completed successfully!")
        return final_results
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(f"❌ {error_msg}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        
        update_progress(model_name, -1, f"Error: {error_msg}")
        
        return {
            "error": error_msg,
            "files_processed": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "traceback": traceback.format_exc()
        }