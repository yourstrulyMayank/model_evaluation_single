import os
import json
import re
import datetime
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

# Advanced metrics
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

HISTORY_FILE = "evaluation_results/llm/tool/bigbench/history.json"

processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress

# --------------------- ADVANCED METRICS --------------------- #
class AdvancedMetrics:
    def __init__(self):
        if ADVANCED_METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
    
    def exact_match(self, prediction: str, target: str) -> float:
        """Normalized exact match."""
        return float(self.normalize(prediction) == self.normalize(target))
    
    def fuzzy_match(self, prediction: str, target: str, threshold: float = 0.8) -> float:
        """Fuzzy string matching using character overlap."""
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
        # Extract first letter/number if it looks like A/B/C/D or 1/2/3/4
        pred_match = re.search(r'^[A-Za-z0-9]', prediction.strip())
        target_match = re.search(r'^[A-Za-z0-9]', target.strip())
        
        if pred_match and target_match:
            return float(pred_match.group().upper() == target_match.group().upper())
        
        return self.exact_match(prediction, target)
    
    def normalize(self, text: str) -> str:
        """Enhanced text normalization."""
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

# --------------------- IMPROVED GENERATION --------------------- #
def generate_response(model, tokenizer, input_text: str, task_type: str, max_new_tokens: int = 128) -> str:
    """Task-aware response generation."""
    device = next(model.parameters()).device
    
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
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_params)
        
        # Extract only the generated part (remove input)
        if hasattr(model, 'config') and 'gpt' in str(type(model)).lower():
            # For GPT-style models, remove the input tokens
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        else:
            generated_ids = output_ids[0]
        
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean up response
        if task_type == 'multiple_choice':
            # Extract first meaningful token for MC questions
            match = re.search(r'^[A-Za-z0-9]', response)
            if match:
                response = match.group()
        
        return response
        
    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        return ""

# --------------------- MAIN EVALUATION FUNCTION --------------------- #
def run_evaluation(model_name: str, num_examples: int = 50, max_new_tokens: int = 128, 
                          use_full_bigbench: bool = False):
    """Enhanced evaluation with multiple metrics and better task handling."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Initialize metrics
    metrics = AdvancedMetrics()
    
    # Load model
    print(f"🔍 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(model_name)
    
    # Smart model loading
    if config.architectures:
        arch = config.architectures[0].lower()
        if any(name in arch for name in ['seq2seq', 't5', 'bart']):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print("📦 Loaded as Seq2Seq model")
        elif any(name in arch for name in ['causallm', 'gpt', 'gemma']):
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print("📦 Loaded as Causal LM")
        else:
            model = AutoModel.from_pretrained(model_name)
            print("📦 Loaded as generic model")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("📦 Loaded as Causal LM (fallback)")
    
    model.to(device)
    model.eval()
    
    # Get tasks
    vocab = vocabs.ALL_VOCABS["t5_default"]
    
    if use_full_bigbench:
        # Use more comprehensive task list
        mixture_names = [
            "bigbench:bigbench_lite_v1.mix.t5_default_vocab.0_shot.1024_examples",
            "bigbench:bigbench_lite_v1.mix.t5_default_vocab.1_shot.1024_examples",
        ]
    else:
        mixture_names = [
            "bigbench:bigbench_lite_v1.mix.t5_default_vocab.0_shot.1024_examples"
        ]
    
    task_names = set()
    for mix_name in mixture_names:
        try:
            mix = seqio.get_mixture_or_task(mix_name)
            task_names.update([t.name for t in mix.tasks])
        except Exception as e:
            print(f"⚠️ Failed to load mixture: {mix_name}: {e}")
    
    task_names = sorted(task_names)
    print(f"📘 Loaded {len(task_names)} unique tasks")
    
    # Evaluation loop
    all_results = []
    task_type_results = defaultdict(list)
    
    for task_name in task_names:
        print(f"\n🔍 Evaluating task: {task_name}")
        print("=" * 100)
        
        task_type = get_task_type(task_name)
        print(f"📋 Task type: {task_type}")
        
        try:
            task = seqio.get_mixture_or_task(task_name)
            dataset = task.get_dataset(split="validation")
            
            task_metrics = defaultdict(list)
            samples = []
            
            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break
                
                input_text = vocab.vocabulary.decode(example["inputs"].numpy())
                target_text = vocab.vocabulary.decode(example["targets"].numpy()).strip()
                
                # Generate response
                prediction = generate_response(model, tokenizer, input_text, task_type, max_new_tokens)
                
                # Evaluate
                eval_results = evaluate_example(prediction, target_text, task_type, metrics)
                
                # Store results
                for metric_name, score in eval_results.items():
                    task_metrics[metric_name].append(score)
                
                sample_data = {
                    "example_number": i + 1,
                    "input": input_text[:200] + "..." if len(input_text) > 200 else input_text,
                    "expected": target_text,
                    "generated": prediction,
                    "metrics": eval_results,
                    "task_type": task_type
                }
                samples.append(sample_data)
                
                # Print progress
                if i % 10 == 0 or i < 5:
                    print(f"🔹 Example {i+1}: Primary={eval_results['primary_metric']:.3f}")
            
            # Aggregate task results
            task_summary = {}
            for metric_name, scores in task_metrics.items():
                task_summary[metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
            
            print(f"\n📊 Task Results for {task_name}:")
            print(f"   Primary Metric: {task_summary.get('primary_metric', {}).get('mean', 0):.3f}")
            print(f"   Exact Match: {task_summary.get('exact_match', {}).get('mean', 0):.3f}")
            
            task_result = {
                "task": task_name,
                "task_type": task_type,
                "summary": task_summary,
                "samples": samples[:5],  # Store only first 5 samples to save space
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            all_results.append(task_result)
            task_type_results[task_type].append(task_summary.get('primary_metric', {}).get('mean', 0))
            
        except Exception as e:
            print(f"❌ Failed to evaluate task '{task_name}': {e}")
    
    # Overall summary
    print(f"\n" + "="*50)
    print(f"📈 EVALUATION SUMMARY")
    print(f"="*50)
    
    for task_type, scores in task_type_results.items():
        if scores:
            print(f"{task_type.upper()}: {np.mean(scores):.3f} ± {np.std(scores):.3f} ({len(scores)} tasks)")
    
    overall_scores = [s for scores in task_type_results.values() for s in scores]
    if overall_scores:
        print(f"OVERALL: {np.mean(overall_scores):.3f} ± {np.std(overall_scores):.3f}")
    
    # Save results
    _save_enhanced_results(model_name, all_results, task_type_results)
    return all_results



def _save_enhanced_results(model_path: str, results: List[Dict], task_type_results: Dict):
    """Save enhanced results with summary statistics."""
    
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
        "model_path": model_path,
        "summary": summary,
        "detailed_results": results,
        "timestamp": datetime.datetime.now().isoformat(),
        "num_tasks": len(results)
    }
    
    # Load existing history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    
    history.insert(0, entry)
    
    # Save
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"💾 Results saved to {HISTORY_FILE}")

    # Save to allbenchmarkhistory.json with correct structure and run numbering
    allbenchhistory_file = "evaluation_results/allbenchmarkhistory.json"
    
    # Load existing history
    allbenchhistory_data = []
    if os.path.exists(allbenchhistory_file):
        try:
            with open(allbenchhistory_file, 'r') as f:
                allbenchhistory_data = json.load(f)
        except:
            allbenchhistory_data = []
    
    # Find the highest run number for this model
    model_runs = [entry.get('run', 0) for entry in allbenchhistory_data 
                  if entry.get('model_path') == model_path]
    next_run = max(model_runs) + 1 if model_runs else 1
    
    # Convert score to percentage for consistency with the template
    benchmark_percentage = benchmark_mean * 100
    
    # Create entry with correct structure that matches your desired format
    allbench_new_entry = {
        'model_path': model_path,
        'benchmark': 'BIG-Bench',        
        'average': benchmark_percentage,
        'run': next_run
    }
    
    allbenchhistory_data.append(allbench_new_entry)

    # Save back to file
    os.makedirs("evaluation_results", exist_ok=True)
    with open(allbenchhistory_file, 'w') as f:
        json.dump(allbenchhistory_data, f, indent=2)
    
    print(f"💾 Benchmark history saved with run number: {next_run}")


def save_benchmark_result(model_name, results, benchmark_name='BIG-Bench', timestamp=None):
    """Save benchmark results to allbenchmarkhistory.json with correct structure and run numbering"""
    if timestamp is None:
        timestamp = datetime.datetime.now().isoformat()
    
    history_file = "evaluation_results/allbenchmarkhistory.json"
    
    # Load existing history
    history_data = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history_data = json.load(f)
        except:
            history_data = []
    
    # Find the highest run number for this model and benchmark combination
    model_benchmark_runs = [entry.get('run', 0) for entry in history_data 
                           if entry.get('model_path') == model_name and 
                              entry.get('benchmark') == benchmark_name]
    next_run = max(model_benchmark_runs) + 1 if model_benchmark_runs else 1
    
    # Handle different result types
    if isinstance(results, dict):
        # If results is a dict, look for average or calculate from scores
        average = results.get('average', 0)
        if average == 0 and 'scores' in results:
            benchmark_scores = [v for v in results['scores'].values() if isinstance(v, (int, float))]
            average = sum(benchmark_scores) / len(benchmark_scores) if benchmark_scores else 0
    else:
        # If results is a single score
        average = results
    
    # Create new entry with correct structure
    new_entry = {
        'model_path': model_name,
        'benchmark': benchmark_name,
        'average': average,
        'run': next_run
    }
    
    # Add to history
    history_data.append(new_entry)
    
    # Save back to file
    os.makedirs("evaluation_results", exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"💾 Benchmark result saved with run number: {next_run}")

def get_history(model_name=None):
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []

    if model_name:
        return [entry for entry in data if model_name in entry["model_path"]]
    return data


def run_evaluation_in_background(model_name, model_path, eval_params):
    """Enhanced background evaluation with progress tracking."""
    processing_status[model_name] = "processing"
    evaluation_progress[model_name] = {
        "current_task": "",
        "completed_tasks": 0,
        "total_tasks": 0,
        "progress_percent": 0
    }

    def background_task():
        try:
            # Update progress
            evaluation_progress[model_name]["current_task"] = "Initializing..."
            
            # Run enhanced evaluation
            result = run_evaluation(
                model_name=model_path,
                num_examples=eval_params.get('num_examples', 25),
                max_new_tokens=eval_params.get('max_tokens', 128),
                use_full_bigbench=eval_params.get('full_benchmark', False)
            )
            
            processing_status[model_name] = "complete"
            evaluation_progress[model_name]["progress_percent"] = 100
            evaluation_progress[model_name]["current_task"] = "Completed"
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            processing_status[model_name] = "error"
            evaluation_progress[model_name]["current_task"] = f"Error: {str(e)}"

    threading.Thread(target=background_task).start()

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