#llm_tool_bigbench_utils.py
import os
import json
import re
from datetime import datetime
import random
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import threading
import torch
# import seqio
import numpy as np
from datasets import disable_caching
# from bigbench.bbseqio import tasks, vocabs
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoConfig
)
# To this:
import threading
from collections import defaultdict
progress_tracker = {}
progress_lock = threading.Lock()
task_completion_tracker = defaultdict(set)
RESULTS_FILE = "evaluation_results.json"

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    import nltk
    nltk.download('punkt', quiet=True)
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced metrics not available. Install: pip install rouge-score nltk scikit-learn")
    ADVANCED_METRICS_AVAILABLE = False

disable_caching()

# HISTORY_FILE = "evaluation_results/llm/tool/bigbench/history.json"

# _evaluation_progress = {}
# processing_status = {}
# from app import evaluation_progress, processing_status

STANDARD_EVAL_STAGES = [
    "Initializing model and tokenizer...",
    "Loading benchmark tasks...",
    "Running evaluation on tasks...",
    "Aggregating results...",
    "Finalizing and saving results..."
]

def save_results_to_file(model_name: str, results_data: dict):
    """Save results to local JSON file."""
    try:
        # Load existing results
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Save new results
        all_results[model_name] = [results_data]
        
        # Write back to file
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✅ Results saved to file for {model_name}")
        return True
    except Exception as e:
        print(f"❌ Error saving results to file: {e}")
        return False

def load_results_from_file(model_name: str = None):
    """Load results from local JSON file."""
    try:
        if not os.path.exists(RESULTS_FILE):
            return {} if model_name is None else []
        
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        
        if model_name:
            return all_results.get(model_name, [])
        return all_results
    except Exception as e:
        print(f"❌ Error loading results from file: {e}")
        return {} if model_name is None else []

def clear_results_file():
    """Clear the results file."""
    try:
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        print("🧹 Results file cleared")
    except Exception as e:
        print(f"❌ Error clearing results file: {e}")


# def update_standard_progress(model_name, stage, message, task_details=None):
#     """Update progress for standard evaluation with enhanced task tracking."""
#     progress_data = {
#         'stage': stage,
#         'message': message,
#         'timestamp': datetime.now().isoformat()
#     }
    
#     # Add task-specific details if provided
#     if task_details:
#         progress_data.update(task_details)
    
#     # Update local progress tracker with thread safety
#     with progress_lock:
#         if model_name not in progress_tracker:
#             progress_tracker[model_name] = {}
#         progress_tracker[model_name].update(progress_data)
    
#     # Also try to update app module if available (optional, non-critical)
#     try:
#         import app
#         app.evaluation_progress[model_name] = progress_data
        
#         # Update processing status based on stage
#         if stage == -1:
#             app.processing_status[model_name] = "error"
#         elif stage >= 5:
#             app.processing_status[model_name] = "complete"
#         else:
#             app.processing_status[model_name] = "processing"
#     except:
#         # Silently fail - local tracking is the primary mechanism
#         pass
    
#     print(f"📊 Standard Progress Update - {model_name}: Stage {stage} - {message}")

def update_task_progress(model_name, task_name, task_index, total_tasks, status="processing"):
    """Update progress for individual task completion."""
    with progress_lock:
        if status == "completed":
            task_completion_tracker[model_name].add(task_name)
        
        completed_count = len(task_completion_tracker[model_name])
        
        # Create a more stable progress message
        if status == "processing":
            message = f"Evaluating tasks... ({completed_count}/{total_tasks} completed) - Current: {task_name[:40]}..."
        else:
            message = f"Evaluating tasks... ({completed_count}/{total_tasks} completed)"
        
        task_details = {
            'current_task': task_name,
            'completed_tasks': completed_count,
            'total_tasks': total_tasks,
            'completion_percentage': int((completed_count / total_tasks) * 100)
        }
        
        update_standard_progress(model_name, 3, message, task_details)



def get_progress(model_name):
    """Get progress for a model with thread safety."""
    with progress_lock:
        return progress_tracker.get(model_name, {'stage': 0, 'message': 'Not started'})

def clear_progress(model_name):
    """Clear progress tracking for a specific model."""
    with progress_lock:
        if model_name in progress_tracker:
            del progress_tracker[model_name]
    print(f"🧹 Cleared progress tracking for {model_name}")


# --------------------- ADVANCED METRICS --------------------- #
class AdvancedMetrics:
    def __init__(self):
        if ADVANCED_METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
    
    def exact_match(self, prediction: str, target: str) -> float:
        """Normalized exact match."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        return float(self.normalize(prediction) == self.normalize(target))
    
    def fuzzy_match(self, prediction: str, target: str, threshold: float = 0.8) -> float:
        """Fuzzy string matching using character overlap."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        pred_norm = self.normalize(prediction)
        target_norm = self.normalize(target)
        
        if not pred_norm or not target_norm:
            return 0.0
        
        # Simple character-level similarity
        common_chars = sum(1 for c in pred_norm if c in target_norm)
        similarity = common_chars / max(len(pred_norm), len(target_norm))
        return float(similarity >= threshold)
    
    def rouge_scores(self, prediction: str, target: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not ADVANCED_METRICS_AVAILABLE:
            return {}
        
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        try:
            scores = self.rouge_scorer.score(target, prediction)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure,
            }
        except:
            return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}
    
    def bleu_score(self, prediction: str, target: str) -> float:
        """Calculate BLEU score."""
        if not ADVANCED_METRICS_AVAILABLE:
            return 0.0
        
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        try:
            pred_tokens = prediction.lower().split()
            target_tokens = [target.lower().split()]  # BLEU expects list of reference lists
            
            if not pred_tokens or not any(target_tokens):
                return 0.0
            
            return sentence_bleu(target_tokens, pred_tokens, smoothing_function=self.smoothing)
        except:
            return 0.0
    
    def token_f1(self, prediction: str, target: str) -> float:
        """Token-level F1 score."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        pred_tokens = set(prediction.lower().split())
        target_tokens = set(target.lower().split())
        
        if not pred_tokens and not target_tokens:
            return 1.0
        if not pred_tokens or not target_tokens:
            return 0.0
        
        common = pred_tokens & target_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(target_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def multiple_choice_accuracy(self, prediction: str, target: str, choices: List[str] = None) -> float:
        """Specialized accuracy for multiple choice questions."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        # Extract first letter/number if it looks like A/B/C/D or 1/2/3/4
        pred_match = re.search(r'^[A-Za-z0-9]', prediction.strip())
        target_match = re.search(r'^[A-Za-z0-9]', target.strip())
        
        if pred_match and target_match:
            return float(pred_match.group().upper() == target_match.group().upper())
        
        return self.exact_match(prediction, target)
    
    def normalize(self, text: str) -> str:
        """Enhanced text normalization."""
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove common punctuation that doesn't affect meaning
        text = re.sub(r'[.,;:!?"\']', '', text)
        return text

# --------------------- TASK-SPECIFIC EVALUATION --------------------- #
def get_task_type(task_name: str) -> str:
    """Infer task type from task name for specialized evaluation."""
    task_lower = task_name.lower()
    
    if any(keyword in task_lower for keyword in ['multiple_choice', 'choice', 'qa']):
        return 'multiple_choice'
    elif any(keyword in task_lower for keyword in ['generation', 'summariz', 'translation']):
        return 'generation'
    elif any(keyword in task_lower for keyword in ['classification', 'sentiment', 'toxic']):
        return 'classification'
    elif any(keyword in task_lower for keyword in ['reasoning', 'logic', 'math']):
        return 'reasoning'
    else:
        return 'general'

def evaluate_example(prediction: str, target: str, task_type: str, metrics: AdvancedMetrics) -> Dict[str, float]:
    """Comprehensive evaluation of a single example."""
    results = {}
    
    # Ensure prediction and target are strings
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else ""
    if not isinstance(target, str):
        target = str(target) if target is not None else ""
    
    # Always compute exact match
    results['exact_match'] = metrics.exact_match(prediction, target)
    
    # Task-specific metrics
    if task_type == 'multiple_choice':
        results['mc_accuracy'] = metrics.multiple_choice_accuracy(prediction, target)
        results['primary_metric'] = results['mc_accuracy']
    elif task_type == 'generation':
        rouge_scores = metrics.rouge_scores(prediction, target)
        results.update(rouge_scores)
        results['bleu'] = metrics.bleu_score(prediction, target)
        results['token_f1'] = metrics.token_f1(prediction, target)
        results['primary_metric'] = rouge_scores.get('rouge1_f', results['exact_match'])
    elif task_type in ['classification', 'reasoning']:
        results['fuzzy_match'] = metrics.fuzzy_match(prediction, target)
        results['token_f1'] = metrics.token_f1(prediction, target)
        results['primary_metric'] = max(results['exact_match'], results['fuzzy_match'])
    else:
        results['token_f1'] = metrics.token_f1(prediction, target)
        results['primary_metric'] = results['exact_match']
    
    return results

def generate_response(model, tokenizer, input_text: str, task_type: str, max_new_tokens: int = 128) -> str:
    """Task-aware response generation with better device handling."""
    
    # Get device from model parameters, with fallback
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
        print("⚠️ Could not determine model device, using CPU")
    
    # Ensure input_text is a string
    if not isinstance(input_text, str):
        input_text = str(input_text) if input_text is not None else ""
    
    # Task-specific generation parameters
    gen_params = {
        'max_new_tokens': max_new_tokens,
        'do_sample': False,  # Deterministic for evaluation
        'pad_token_id': tokenizer.eos_token_id,
    }
    
    if task_type == 'multiple_choice':
        gen_params.update({
            'max_new_tokens': min(10, max_new_tokens),  # Short responses for MC
            'temperature': 0.1,
        })
    elif task_type == 'generation':
        gen_params.update({
            'temperature': 0.3,
            'do_sample': True,
            'top_p': 0.9,
        })
    
    try:
        # Tokenize input with proper error handling
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        )
        
        # Move inputs to the same device as model
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as device_error:
            print(f"⚠️ Could not move inputs to {device}, keeping on CPU: {device_error}")
            # Keep inputs on CPU if device movement fails
        
        with torch.no_grad():
            try:
                output_ids = model.generate(**inputs, **gen_params)
            except Exception as gen_error:
                print(f"⚠️ Generation failed with device {device}, trying CPU: {gen_error}")
                # Fallback: move everything to CPU
                if device != torch.device("cpu"):
                    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                    model_cpu = model.cpu()
                    output_ids = model_cpu.generate(**inputs_cpu, **gen_params)
                    # Move model back to original device if possible
                    try:
                        model.to(device)
                    except:
                        pass
                else:
                    raise gen_error
        
        # Extract only the generated part (remove input)
        if hasattr(model, 'config') and 'gpt' in str(type(model)).lower():
            # For GPT-style models, remove the input tokens
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        else:
            generated_ids = output_ids[0]
        
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response) if response is not None else ""
        
        response = response.strip()
        
        # Clean up response
        if task_type == 'multiple_choice':
            # Extract first meaningful token for MC questions
            match = re.search(r'^[A-Za-z0-9]', response)
            if match:
                response = match.group()
        
        return response
        
    except Exception as e:
        print(f"⚠️ Generation failed completely: {e}")
        return ""

def run_evaluation(model_name: str, model_path: str = None, num_examples: int = 50, max_new_tokens: int = 128, 
                          use_full_bigbench: bool = False):
    """Sequential evaluation with immediate progress updates."""
    
    if model_path is None:
        model_path = model_name
    
    try:
        # Force CPU and disable GPU warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import warnings
        warnings.filterwarnings('ignore')
        
        # Stage 1: Load model
        update_standard_progress(model_name, 1, "Loading model and tokenizer...")
        print(f"📊 Stage 1: Loading model for {model_name}")
        
        device = torch.device("cpu")
        metrics = AdvancedMetrics()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        if config.architectures:
            arch = config.architectures[0].lower()
            if any(name in arch for name in ['seq2seq', 't5', 'bart']):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Handle meta tensors properly
        try:
            model = model.to(device)
        except RuntimeError as e:
            if "meta tensor" in str(e):
                print("🔧 Handling meta tensors with to_empty()")
                model = model.to_empty(device=device)
                # Initialize parameters if they're meta tensors
                for name, param in model.named_parameters():
                    if param.is_meta:
                        # Initialize with small random values
                        param.data = torch.randn_like(param, device=device) * 0.02
                for name, buffer in model.named_buffers():
                    if buffer.is_meta:
                        buffer.data = torch.zeros_like(buffer, device=device)
            else:
                raise e
        
        model.eval()
        print(f"✅ Model loaded on {device}")
        
        # IMMEDIATELY advance to Stage 2
        update_standard_progress(model_name, 2, "Loading benchmark tasks...")
        print(f"📊 Stage 2: Loading tasks for {model_name}")
        
        vocab = vocabs.ALL_VOCABS["t5_default"]
        mix_name = "bigbench:bigbench_lite_v1.mix.t5_default_vocab.0_shot.1024_examples"
        mix = seqio.get_mixture_or_task(mix_name)
        task_names = sorted([t.name for t in mix.tasks])
        task_names = task_names[:20]  # Limit for testing
        print(f"✅ Loaded {len(task_names)} tasks")
        
        # IMMEDIATELY advance to Stage 3
        update_standard_progress(model_name, 3, f"Running evaluation on {len(task_names)} tasks...")
        print(f"📊 Stage 3: Starting evaluation for {model_name}")
        
        all_results = []
        task_type_results = defaultdict(list)
        
        # Sequential task processing - NO THREADING
        for task_idx, task_name in enumerate(task_names):
            current_task_msg = f"Evaluating task {task_idx + 1}/{len(task_names)}: {task_name[:30]}..."
            update_standard_progress(model_name, 3, current_task_msg)
            print(f"🔍 {current_task_msg}")
            
            task_type = get_task_type(task_name)
            task = seqio.get_mixture_or_task(task_name)
            dataset = task.get_dataset(split="validation")
            
            task_metrics = defaultdict(list)
            samples = []
            
            # Sequential example processing
            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break
                
                try:
                    input_text = vocab.vocabulary.decode(example["inputs"].numpy())
                    target_text = vocab.vocabulary.decode(example["targets"].numpy()).strip()
                    
                    prediction = generate_response(model, tokenizer, input_text, task_type, max_new_tokens)
                    eval_results = evaluate_example(prediction, target_text, task_type, metrics)
                    
                    for metric_name, score in eval_results.items():
                        task_metrics[metric_name].append(score)
                    
                    sample_data = {
                        "example_number": i + 1,
                        "input": input_text[:200],
                        "expected": target_text,
                        "generated": prediction,
                        "metrics": eval_results,
                        "task_type": task_type
                    }
                    samples.append(sample_data)
                    
                except Exception as e:
                    print(f"⚠️ Skipping example {i}: {e}")
                    continue
            
            # Process task results
            task_summary = {}
            for metric_name, scores in task_metrics.items():
                if scores:
                    task_summary[metric_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'count': len(scores)
                    }
            
            task_result = {
                "task": task_name,
                "task_type": task_type,
                "summary": task_summary,
                "samples": samples[:5],
                "timestamp": datetime.now().isoformat()
            }
            
            all_results.append(task_result)
            if 'primary_metric' in task_summary:
                task_type_results[task_type].append(task_summary['primary_metric']['mean'])
            
            print(f"✅ Completed task {task_idx + 1}/{len(task_names)}")
        
        # IMMEDIATELY advance to Stage 4 after all tasks complete
        update_standard_progress(model_name, 4, "Aggregating results...")
        print(f"📊 Stage 4: Aggregating results for {model_name}")
        
        summary = {}
        for task_type, scores in task_type_results.items():
            if scores:
                summary[task_type] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores)
                }
        
        overall_scores = [s for scores in task_type_results.values() for s in scores]
        if overall_scores:
            summary['overall'] = {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'count': len(overall_scores)
            }
        
        entry = {
            "model_path": model_name,
            "summary": summary,
            "detailed_results": all_results,
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(all_results),
            "status": "completed"
        }
        
        # IMMEDIATELY advance to Stage 5
        update_standard_progress(model_name, 5, "Saving results...")
        print(f"📊 Stage 5: Saving results for {model_name}")
        
        # Save to file instead of trying to import app
        if save_results_to_file(model_name, entry):
            print(f"✅ Results saved to file for {model_name}")
        else:
            raise Exception("Failed to save results to file")
        
        return all_results
    
    except Exception as e:
        update_standard_progress(model_name, -1, f"Error: {str(e)}")
        print(f"❌ Evaluation failed for {model_name}: {e}")
        try:
            import app
            app.processing_status[model_name] = "error"
        except:
            pass
        raise e

def run_evaluation_in_background(model_name, model_path, eval_params):
    """Run evaluation in background thread - but evaluation itself is sequential."""
    
    print(f"🚀 Starting evaluation for {model_name}")
    
    # Set initial status
    try:
        import app
        app.processing_status[model_name] = "processing"
        if model_name in app.current_results:
            del app.current_results[model_name]
    except:
        pass

    def background_task():
        try:
            # Run SEQUENTIAL evaluation (no internal threading)
            run_evaluation(
                model_name=model_name,
                model_path=model_path,
                num_examples=eval_params.get('num_examples', 5),
                max_new_tokens=eval_params.get('max_tokens', 128),
                use_full_bigbench=eval_params.get('full_benchmark', False)
            )
                
        except Exception as e:
            print(f"❌ Background evaluation failed for {model_name}: {e}")

    # Only this part uses threading - the evaluation itself is sequential
    threading.Thread(target=background_task, daemon=True).start()


def update_standard_progress(model_name, stage, message, task_details=None):
    """Simple, immediate progress update."""
    progress_data = {
        'stage': stage,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if task_details:
        progress_data.update(task_details)
    
    # Update local progress tracker
    with progress_lock:
        progress_tracker[model_name] = progress_data
    
    # Update app immediately
    try:
        import app
        app.evaluation_progress[model_name] = progress_data
        
        # Update status based on stage
        if stage == -1:
            app.processing_status[model_name] = "error"
        elif stage >= 5:  # Change this back to 5 instead of 6
            app.processing_status[model_name] = "complete"
        else:
            app.processing_status[model_name] = "processing"
            
        print(f"📊 Progress updated: {model_name} -> Stage {stage}: {message}")
    except Exception as e:
        print(f"⚠️ Failed to update app progress: {e}")

def _save_enhanced_results(model_name: str, results: List[Dict], task_type_results: Dict):
    """Save current results only AFTER evaluation is complete."""
    
    summary = {}
    for task_type, scores in task_type_results.items():
        if scores:
            summary[task_type] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores),
                'scores': scores
            }
    
    overall_scores = [s for scores in task_type_results.values() for s in scores]
    benchmark_mean = float(np.mean(overall_scores)) if overall_scores else 0.0
    if overall_scores:
        summary['overall'] = {
            'mean': benchmark_mean,
            'std': float(np.std(overall_scores)),
            'count': len(overall_scores)
        }
    
    entry = {
        "model_path": model_name,
        "summary": summary,
        "detailed_results": results,
        "timestamp": datetime.now().isoformat(),
        "num_tasks": len(results),
        "status": "completed"  # Add explicit completion status
    }
    
    # Store in app's current_results ONLY when evaluation is complete
    try:
        import sys
        if 'app' in sys.modules:
            app = sys.modules['app']
            # Use model_name directly as the key
            app.current_results[model_name] = [entry]
            print(f"💾 FINAL Results stored in memory for {model_name}")
        else:
            print(f"💾 App module not found, results not stored")
    except Exception as e:
        print(f"Failed to store results in app: {e}")


def extract_score_from_results(results):
    """Extract a score from various result formats."""
    try:
        # Handle different possible result structures
        if isinstance(results, dict):
            # Look for common score fields
            score_fields = ['accuracy', 'score', 'exact_match', 'f1', 'bleu', 'rouge_l']
            
            for field in score_fields:
                if field in results:
                    value = results[field]
                    if isinstance(value, (int, float)):
                        return value * 100 if value <= 1.0 else value
                    elif isinstance(value, dict) and 'mean' in value:
                        mean_val = value['mean']
                        return mean_val * 100 if mean_val <= 1.0 else mean_val
            
            # If no direct score field, look for nested structures
            if 'metrics' in results:
                return extract_score_from_results(results['metrics'])
            
            if 'summary' in results:
                return extract_score_from_results(results['summary'])
        
        elif isinstance(results, (int, float)):
            return results * 100 if results <= 1.0 else results
    
    except:
        pass
    
    return None

# --------------------- ENTRY POINT --------------------- #
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced BIG-bench evaluation")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--num_examples", type=int, default=50, help="Examples per task")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max generation tokens")
    parser.add_argument("--full_benchmark", action="store_true", help="Use full BIG-bench (not just lite)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_name=args.model,
        num_examples=args.num_examples,
        max_new_tokens=args.max_tokens,
        use_full_bigbench=args.full_benchmark
    )