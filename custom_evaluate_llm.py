# new custom_evaluate_llm.py

import os
import pandas as pd
from PIL import Image
from flask import current_app
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import json
from datetime import datetime
import traceback
import threading
# Global progress tracker
progress_tracker = {}
progress_lock = threading.Lock()


# Load models
print("🔄 Loading Donut and SentenceTransformer models...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

device = "cuda" if torch.cuda.is_available() else "cpu"
donut_model.to(device)
print(f"✅ Models loaded successfully on device: {device}")


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
    
def extract_answer_from_image(image: Image.Image, prompt: str) -> str:
    """Use Donut to extract an answer from image given a prompt."""
    print(f"📄 Processing prompt: {prompt}")
    image = image.convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    task_prompt = f"<s_docvqa><s_question>{prompt}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    outputs = donut_model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"🔍 Extracted result: {result[:100]}...")
    return result


def clean_number_string(s):
    if isinstance(s, str):
        return s.replace(',', '').strip()
    return s


def is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def compute_confidence(actual, extracted):
    """Compute confidence score between actual and extracted values."""
    if pd.isna(actual) or pd.isna(extracted):
        return 0.0

    actual_str = str(actual).strip()
    extracted_str = str(extracted).strip()
    actual_clean = clean_number_string(actual_str)
    extracted_clean = clean_number_string(extracted_str)

    if is_number(actual_clean) and is_number(extracted_clean):
        actual_num = float(actual_clean)
        extracted_num = float(extracted_clean)
        denom = max(abs(actual_num), abs(extracted_num), 1)
        similarity = 1 - abs(actual_num - extracted_num) / denom
        score = round(max(0.0, similarity) * 100, 2)
        print(f"🔢 Numerical comparison: {actual_num} vs {extracted_num} = {score}%")
        return score

    actual_emb = embedding_model.encode(actual_str)
    extracted_emb = embedding_model.encode(extracted_str)
    score = cosine_similarity([actual_emb], [extracted_emb])[0][0]
    final_score = round(score * 100, 2)
    print(f"📝 Text similarity: '{actual_str}' vs '{extracted_str}' = {final_score}%")
    return final_score


def grade_confidence(score):
    """Grade the confidence score."""
    if score >= 90:
        return '✅ Pass'
    elif score >= 70:
        return '⚠ Intermittent'
    else:
        return '❌ Fail'


def generate_transaction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate transaction summary analysis."""
    print("📊 Generating transaction summary...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Withdrawls'] = pd.to_numeric(df['Withdrawls'], errors='coerce').fillna(0)
    df['Deposits'] = pd.to_numeric(df['Deposits'], errors='coerce').fillna(0)
    df['month'] = df['Date'].dt.to_period('M')
    df['week'] = df['Date'].dt.to_period('W')
    df['day'] = df['Date'].dt.date

    monthly_deposits = df.groupby('month')['Deposits'].sum().to_dict()
    monthly_withdrawls = df.groupby('month')['Withdrawls'].sum().to_dict()

    prompt_answer_pairs = {
        "Weekly_average_withdrawal_amount?": df.groupby('week')['Withdrawls'].sum().mean(),
        "Daily_average_withdrawal_amount?": df.groupby('day')['Withdrawls'].sum().mean(),
        "Monthly_average_withdrawal_amount?": df.groupby('month')['Withdrawls'].sum().mean(),

        "Weekly_average_deposit_amount?": df.groupby('week')['Deposits'].sum().mean(),
        "Daily_average_deposit_amount?": df.groupby('day')['Deposits'].sum().mean(),
        "Monthly_average_deposit_amount?": df.groupby('month')['Deposits'].sum().mean(),

        "maximum_withdrawal_made?": df.loc[df['Withdrawls'].idxmax(), 'Date'].strftime('%Y-%m-%d'),
        "minimum_withdrawal_made?": df.loc[df['Withdrawls'][df['Withdrawls'] > 0].idxmin(), 'Date'].strftime('%Y-%m-%d'),

        "total_deposit_amount for each_month?": ", ".join([f"{k}: {v:.2f}" for k, v in monthly_deposits.items()]),
        "total_withdrawal_amount for each_month?": ", ".join([f"{k}: {v:.2f}" for k, v in monthly_withdrawls.items()]),

        "Which description has the highest_wihdrawal?": df.loc[df['Withdrawls'].idxmax(), 'Description'],
        "Which description has the lowest_wihdrawal?": df.loc[df['Withdrawls'][df['Withdrawls'] > 0].idxmin(), 'Description'],
    }

    print(f"✅ Generated {len(prompt_answer_pairs)} transaction analysis prompts")
    return pd.DataFrame(list(prompt_answer_pairs.items()), columns=['Prompt', 'Extracted'])


def match_and_merge_results(merged_df, ground_truth_df):
    """Match and merge results with ground truth using fuzzy matching."""
    print("🔍 Matching results with ground truth...")
    ground_truth_df['Prompt_clean'] = ground_truth_df['Prompt'].str.strip().str.lower()
    merged_df['Prompt_clean'] = merged_df['Prompt'].str.strip().str.lower()

    matched_prompts = []
    for actual_prompt in ground_truth_df['Prompt_clean']:
        match, score, _ = process.extractOne(actual_prompt, merged_df['Prompt_clean'], scorer=fuzz.token_sort_ratio)
        matched_prompts.append({'Prompt_clean': actual_prompt, 'Matched_prompt_clean': match, 'Score': score})
        print(f"🎯 Matched '{actual_prompt}' -> '{match}' (score: {score})")

    match_df = pd.DataFrame(matched_prompts)
    gt_matched = ground_truth_df.merge(match_df, on='Prompt_clean', how='left')

    merged_with_extracted = gt_matched.merge(
        merged_df[['Prompt_clean', 'Extracted']],
        left_on='Matched_prompt_clean',
        right_on='Prompt_clean',
        how='left'
    )

    final_df = merged_with_extracted[['Prompt', 'Actual', 'Extracted']]
    final_df['Match_Status'] = merged_with_extracted['Score'].apply(lambda x: '✅' if x >= 90 else '⚠')
    print(f"✅ Matched and merged {len(final_df)} results")
    return final_df


def evaluate_image_and_transactions(image_path, transaction_excel, ground_truth_excel):
    """Main evaluation function for image and transaction analysis."""
    print(f"🚀 Starting evaluation for image: {image_path}")
    
    # Load and process image
    image = Image.open(image_path)
    prompts = [
        "What is the name of the bank?",
        "What is the statement period?",
        "What is the account number?",
        "What is the total withdrawal amount?",
        "What is the total deposit amount?",
        "What is the maximum withdrawals amount?",
        "What is the minimum withdrawals amount?",
        "What is the maximum deposits amount?",
        "What is the minimum deposits amount?"
    ]

    print(f"🔄 Processing {len(prompts)} image prompts...")
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"📄 Processing prompt {i}/{len(prompts)}: {prompt}")
        answer = extract_answer_from_image(image, prompt)
        if answer.lower().startswith(prompt.lower()):
            answer = answer[len(prompt):].strip().strip(":-")
        results.append({"Prompt": prompt, "Extracted": answer})

    image_df = pd.DataFrame(results)
    print(f"✅ Completed image analysis with {len(image_df)} results")

    # Process transaction data
    print(f"📊 Loading transaction data from: {transaction_excel}")
    trans_df = pd.read_excel(transaction_excel)
    print(f"📈 Loaded {len(trans_df)} transaction records")
    
    trans_summary = generate_transaction_summary(trans_df)

    # Merge results
    merged_df = pd.concat([image_df, trans_summary], ignore_index=True)
    print(f"🔗 Merged datasets: {len(merged_df)} total results")

    # Load ground truth
    print(f"📋 Loading ground truth from: {ground_truth_excel}")
    gt_df = pd.read_excel(ground_truth_excel)
    print(f"✅ Loaded {len(gt_df)} ground truth entries")

    # Match and evaluate
    final_df = match_and_merge_results(merged_df, gt_df)
    
    print("🧮 Computing confidence scores...")
    final_df['Confidence_Score'] = final_df.apply(
        lambda r: compute_confidence(r['Actual'], r['Extracted']), axis=1
    )
    final_df['Test_case'] = final_df['Confidence_Score'].apply(grade_confidence)
    final_df['Confidence_Score'] = final_df['Confidence_Score'].round(2)

    print("✅ Evaluation completed successfully!")
    return final_df[['Prompt', 'Actual', 'Extracted', 'Confidence_Score', 'Test_case']]


def find_files_in_directory(upload_dir):
    """Find required files in the upload directory."""
    print(f"🔍 Scanning upload directory: {upload_dir}")
    
    image_files = []
    transaction_files = []
    ground_truth_files = []
    
    if not os.path.exists(upload_dir):
        print(f"❌ Upload directory does not exist: {upload_dir}")
        return None, None, None
    
    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        if os.path.isfile(filepath):
            lower_filename = filename.lower()
            
            # Image files
            if any(ext in lower_filename for ext in ['.png', '.jpg', '.jpeg']):
                image_files.append(filepath)
                print(f"📸 Found image file: {filename}")
            
            # Transaction files
            elif any(keyword in lower_filename for keyword in ['transaction', 'trans']) and \
                 any(ext in lower_filename for ext in ['.xlsx', '.xls', '.csv']):
                transaction_files.append(filepath)
                print(f"📊 Found transaction file: {filename}")
            
            # Ground truth files
            elif any(keyword in lower_filename for keyword in ['ground', 'truth', 'gt']) and \
                 any(ext in lower_filename for ext in ['.xlsx', '.xls', '.csv']):
                ground_truth_files.append(filepath)
                print(f"📋 Found ground truth file: {filename}")
    
    # Return first found file of each type
    image_file = image_files[0] if image_files else None
    transaction_file = transaction_files[0] if transaction_files else None
    ground_truth_file = ground_truth_files[0] if ground_truth_files else None
    
    print(f"📁 File selection summary:")
    print(f"   Image: {os.path.basename(image_file) if image_file else 'None'}")
    print(f"   Transaction: {os.path.basename(transaction_file) if transaction_file else 'None'}")
    print(f"   Ground Truth: {os.path.basename(ground_truth_file) if ground_truth_file else 'None'}")
    
    return image_file, transaction_file, ground_truth_file



def run_custom_evaluation(model_name, model_path, upload_dir):
    """
    Main function to run custom evaluation - this is what app.py imports and calls.
    """
    print(f"🚀 Starting custom evaluation for model: {model_name}")
    print(f"📂 Model path: {model_path}")
    print(f"📁 Upload directory: {upload_dir}")
    
    try:
        # Stage 1: Loading models and initializing
        update_progress(model_name, 1, "Loading models and initializing...")
        
        # Use fixed file paths instead of dynamic discovery
        image_file = os.path.join(upload_dir,'bank_2.png')
        transaction_file = os.path.join(upload_dir,'transaction details.xlsx')
        ground_truth_file = os.path.join(upload_dir,'Ground_truth_data.xlsx')
        
        # Validate required files exist
        missing_files = []
        if not os.path.exists(image_file):
            missing_files.append(f"Image file: {image_file}")
        if not os.path.exists(transaction_file):
            missing_files.append(f"Transaction file: {transaction_file}")
        if not os.path.exists(ground_truth_file):
            missing_files.append(f"Ground truth file: {ground_truth_file}")
        
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            print(f"❌ {error_msg}")
            update_progress(model_name, 1, f"Error: {error_msg}")
            return {
                "error": error_msg,
                "files_processed": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Stage 2: Processing uploaded files
        update_progress(model_name, 2, "Processing uploaded files...")
        print("✅ All required files found, starting evaluation...")
        
        # Stage 3: Analyzing document content
        update_progress(model_name, 3, "Analyzing document content...")
        
        # Run evaluation
        evaluation_df = evaluate_image_and_transactions(
            image_file, 
            transaction_file, 
            ground_truth_file
        )
        
        # Stage 4: Comparing with ground truth
        update_progress(model_name, 4, "Comparing with ground truth...")
        print("📊 Processing evaluation results...")
        
        # Calculate summary statistics
        total_tests = len(evaluation_df)
        pass_count = len(evaluation_df[evaluation_df['Test_case'] == '✅ Pass'])
        intermittent_count = len(evaluation_df[evaluation_df['Test_case'] == '⚠ Intermittent'])
        fail_count = len(evaluation_df[evaluation_df['Test_case'] == '❌ Fail'])
        avg_score = evaluation_df['Confidence_Score'].mean()
        overall_score = avg_score
        
        print(f"📈 Evaluation Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {pass_count} ({pass_count/total_tests*100:.1f}%)")
        print(f"   Intermittent: {intermittent_count} ({intermittent_count/total_tests*100:.1f}%)")
        print(f"   Failed: {fail_count} ({fail_count/total_tests*100:.1f}%)")
        print(f"   Average Score: {avg_score:.1f}%")
        
        # Stage 5: Finalizing
        update_progress(model_name, 5, "Finalizing evaluation...")
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        ground_truth_comparison = []
        for _, row in evaluation_df.iterrows():
            ground_truth_comparison.append({
                'prompt': row['Prompt'],
                'actual': str(row['Actual']),
                'extracted': str(row['Extracted']),
                'score': float(row['Confidence_Score']),
                'grade': row['Test_case']
            })
        
        # Prepare comprehensive results
        results = {
            "model_name": model_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_processed": 3,  # image, transaction, ground truth
            "overall_score": float(overall_score),
            "total_tests": total_tests,
            "pass_count": pass_count,
            "intermittent_count": intermittent_count,
            "fail_count": fail_count,
            "average_score": float(avg_score),
            "success_rate": float(pass_count / total_tests * 100),
            "ground_truth_comparison": ground_truth_comparison,
            "file_info": {
                "image_file": os.path.basename(image_file),
                "transaction_file": os.path.basename(transaction_file),
                "ground_truth_file": os.path.basename(ground_truth_file)
            },
            "summary_statistics": {
                "highest_score": float(evaluation_df['Confidence_Score'].max()),
                "lowest_score": float(evaluation_df['Confidence_Score'].min()),
                "median_score": float(evaluation_df['Confidence_Score'].median()),
                "std_deviation": float(evaluation_df['Confidence_Score'].std())
            }
        }
        
        # Mark as completed
        update_progress(model_name, 6, "Evaluation completed successfully!")
        
        print("🎉 Custom evaluation completed successfully!")
        return results
        
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


# Additional utility functions for compatibility
def get_custom_evaluation_history(model_name):
    """Get custom evaluation history for a model."""
    results_dir = "evaluation_results"
    results_file = os.path.join(results_dir, f"{model_name}_custom_evaluation.json")
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading custom evaluation history: {e}")
            return {}
    
    return {}


def save_evaluation_results(model_name, results):
    """Save evaluation results to file."""
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"{model_name}_custom_evaluation.json")
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved successfully to {results_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False


if __name__ == "__main__":
    # Test function - can be used for standalone testing
    print("🧪 Running custom evaluation test...")
    
    # Example usage:
    # results = run_custom_evaluation("test_model", "/path/to/model", "/path/to/uploads")
    # print(json.dumps(results, indent=2))