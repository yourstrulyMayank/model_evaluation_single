# app.py - Enhanced Flask App
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory
import os
import threading
import json
from datetime import datetime
from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback
import logging
import openpyxl
from collections import defaultdict
import contextlib
import threading
import uuid
from weasyprint import HTML, CSS

from llm_tool_bigbench_utils import ( run_evaluation_in_background,
                                        extract_score_from_results)


import numpy as np




# Add logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




app = Flask(__name__)


app.secret_key = 'your-secret-key-here'
model_base_path = "models"
processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Add this global variable with other globals

MODELS = {
    "Wealth Advisory Model": "models/wealth_advisory", 
    "Compliance Model": "models/compliance"
}
current_results = {}

# Create folders


js_function = '''
<script>
function setBenchmark(type, index) {
    const dropdown = document.getElementById('benchmark-' + type + '-' + index);
    const hiddenInput = document.getElementById('benchmark-input-' + type + '-' + index);
    if (dropdown && hiddenInput) {
        hiddenInput.value = dropdown.value;
    }
}
</script>
'''

STANDARD_EVAL_STAGES = [
    "Initializing model and tokenizer...",
    "Loading benchmark tasks...",
    "Running evaluation on tasks...",
    "Aggregating results...",
    "Finalizing and saving results..."
]

# def update_standard_progress(model_name, stage, message):
#     """Update progress for standard evaluation with proper synchronization."""
#     progress_data = {
#         'stage': stage,
#         'message': message,
#         'timestamp': datetime.now().isoformat()
#     }
    
#     # Store locally first
#     _evaluation_progress[model_name] = progress_data
    
#     # Try to update app module - this is the key fix
#     try:
#         import sys
#         if 'app' in sys.modules:
#             app = sys.modules['app']
#             if hasattr(app, 'evaluation_progress'):
#                 app.evaluation_progress[model_name] = progress_data
#             if hasattr(app, 'processing_status'):
#                 # Update processing status based on stage
#                 if stage == -1:
#                     app.processing_status[model_name] = "error"
#                 elif stage >= 6:
#                     app.processing_status[model_name] = "complete"
#                 else:
#                     app.processing_status[model_name] = "processing"
#             print(f"📊 Standard Progress Update - {model_name}: Stage {stage} - {message}")
#         else:
#             print(f"📊 Progress Update (app module not found) - {model_name}: Stage {stage} - {message}")
#     except Exception as e:
#         print(f"Failed to update app progress: {e}")
#         print(f"📊 Progress Update (local fallback) - {model_name}: Stage {stage} - {message}")


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/download_report/<model_name>')
def download_report(model_name):
    """Generate PDF report from current results only."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:
        from jinja2 import Template
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
    
    if not PDF_AVAILABLE:
        flash("PDF generation not available. Please install weasyprint.")
        return redirect(url_for('analyze', model_name=model_name))
    
    try:
        # Get current results only
        results_data = current_results.get(model_name, [])
        
        if not results_data:
            flash("No evaluation results found for this model.")
            return redirect(url_for('analyze', model_name=model_name))
        
        # Render HTML template for PDF
        html_content = render_template('pdf_report.html', 
                                     model_name=model_name, 
                                     results=results_data)
        
        # Generate PDF
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        # Create response
        response = make_response(pdf_buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_evaluation_report.pdf"'
        
        return response
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        flash(f"Error generating PDF report: {str(e)}")
        return redirect(url_for('analyze', model_name=model_name))

@app.route('/export_json/<model_name>')
def export_json(model_name):
    """Export current results as JSON file."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:
        results_data = current_results.get(model_name, [])
        
        if not results_data:
            flash("No evaluation results found for this model.")
            return redirect(url_for('analyze', model_name=model_name))
        
        # Create JSON response
        response = make_response(json.dumps(results_data, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_results.json"'
        
        return response
        
    except Exception as e:
        flash(f"Error exporting JSON: {str(e)}")
        return redirect(url_for('analyze', model_name=model_name))

    
@app.route('/evaluate_model/<model_name>')
def evaluate(model_name):
    # Simplified - no category needed
    if model_name not in MODELS:
        return f"Model '{model_name}' not found.", 404

    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        return f"Model '{model_name}' not found.", 404

    # Set initial status
    processing_status[model_name] = "processing"
    evaluation_progress[model_name] = {
        'stage': 0,
        'message': 'Preparing to start evaluation...',
        'timestamp': datetime.now().isoformat()
    }

    return render_template('loading.html', model_name=model_name)

@app.route('/start_evaluation/<model_name>', methods=['POST'])
def start_evaluation(model_name):
    if model_name not in MODELS:
        return jsonify({'error': 'Unknown model'}), 400

    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        return jsonify({'error': f"Model '{model_name}' not found."}), 404

    # Default evaluation parameters
    eval_params = {
        'num_examples': 25,
        'max_tokens': 128,
        'full_benchmark': False
    }
    
    # Start evaluation in background
    run_evaluation_in_background(model_name, model_path, eval_params)
    
    return jsonify({'status': 'started', 'message': 'Evaluation started successfully'})


# @app.route('/evaluate_llm/<model_name>', methods=['POST', 'GET'])
# def evaluate_llm(model_name):
#     if request.method == 'GET':
#         # Just show the loading page
#         processing_status[model_name] = "processing"
#         evaluation_progress[model_name] = {
#             'stage': 0,
#             'message': 'Preparing to start evaluation...',
#             'timestamp': datetime.now().isoformat()
#         }
#         return render_template('loading.html', model_name=model_name)
    
#     # Handle POST request (actual evaluation)
#     benchmark = "BIG-Bench"
#     num_examples = int(request.form.get('num_examples', 25))
#     max_tokens = int(request.form.get('max_tokens', 128))
#     full_benchmark = request.form.get('full_benchmark') == 'on'

#     print(f"Evaluating {model_name} on benchmark: {benchmark}")

#     # Determine folder based on model_name
#     if model_name == "Wealth Advisory Model":
#         model_folder = "wealth_advisory"
#     elif model_name == "Compliance Model":
#         model_folder = "compliance"
#     else:
#         flash(f"Unknown model: {model_name}")
#         return redirect(url_for('index'))

#     model_path = os.path.join(model_base_path, model_folder)
#     if not os.path.exists(model_path):
#         return f"Model '{model_name}' not found in '{model_folder}'.", 404

#     eval_params = {
#         'num_examples': num_examples,
#         'max_tokens': max_tokens,
#         'full_benchmark': full_benchmark
#     }

#     run_evaluation_in_background(model_name, model_path, eval_params)
#     return render_template('loading.html', model_name=model_name)
# Add this global variable with other globals
current_results = {}

# Updated evaluate_llm function
@app.route('/evaluate_llm/<model_name>', methods=['POST', 'GET'])
def evaluate_llm(model_name):
    if request.method == 'GET':
        # Just show the loading page
        processing_status[model_name] = "processing"
        evaluation_progress[model_name] = {
            'stage': 0,
            'message': 'Preparing to start evaluation...',
            'timestamp': datetime.now().isoformat()
        }
        return render_template('loading.html', model_name=model_name)
    
    # Handle POST request (actual evaluation)
    print(f"Evaluating {model_name}")

    # Check if model exists in MODELS
    if model_name not in MODELS:
        flash(f"Unknown model: {model_name}")
        return redirect(url_for('index'))

    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        return f"Model '{model_name}' not found.", 404

    # Default evaluation parameters
    eval_params = {
        'num_examples': 5,
        'max_tokens': 128,
        'full_benchmark': False
    }

    run_evaluation_in_background(model_name, model_path, eval_params)
    return render_template('loading.html', model_name=model_name)

@app.route('/check_status/<model_name>')
def check_status(model_name):
    # Get app-level status (fallback to not_started)
    status = processing_status.get(model_name, "not_started")
    
    # Get progress from utils module (primary source)
    try:
        from llm_tool_bigbench_utils import get_progress
        progress = get_progress(model_name)
        
        # Sync progress back to app for consistency
        evaluation_progress[model_name] = progress
        
    except Exception as e:
        print(f"⚠️ Could not get progress from utils: {e}")
        # Fallback to app's local progress
        progress = evaluation_progress.get(model_name, {'stage': 0, 'message': 'Not started'})
    
    # Auto-update status based on progress stage if needed
    if progress.get('stage', 0) >= 6 and status != "complete":
        status = "complete" 
        processing_status[model_name] = "complete"
    elif progress.get('stage', 0) == -1 and status != "error":
        status = "error"
        processing_status[model_name] = "error"
    elif progress.get('stage', 0) > 0 and status == "not_started":
        status = "processing"
        processing_status[model_name] = "processing"
    
    # Debug logging
    print(f"📊 Status check for {model_name}: status={status}, progress_stage={progress.get('stage', 0)}")
    
    return jsonify({
        "status": status,
        "progress": progress,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/history/<category>/<model_name>')
def history(category, model_name):
    """Display benchmark history for a specific model."""
    try:
        # Load history data
        history_file = "evaluation_results/llm/allbenchmarkhistory.json"
        history_data = []
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                all_history = json.load(f)
            
            # Filter data for this specific model
            model_history = [entry for entry in all_history if model_name in entry.get("model_path", "")]
            
            # Sort by run number
            model_history.sort(key=lambda x: x.get("run", 0))
            
            # Process the history data for the template
            history_data = []
            for entry in model_history:
                processed_entry = {
                    'run': f"Run {entry.get('run', 1)}",
                    'scores': {entry.get('benchmark', 'BIG-Bench'): entry.get('average', 0)},
                    'average': entry.get('average', 0)
                }
                history_data.append(processed_entry)
            
        # Define all benchmarks we want to track
        benchmark_list = [
            "MMLU", "HellaSwag", "PIQA", "SocialIQA", "BooIQ", 
            "WinoGrande", "CommonsenseQA", "OpenBookQA", "ARC-e", 
            "ARC-c", "TriviaQA", "Natural Questions", "HumanEval", 
            "MBPP", "GSM8K", "MATH", "AGIEval", "BIG-Bench"
        ]
        
        # Calculate benchmark averages
        benchmark_averages = {}
        benchmark_counts = defaultdict(int)
        benchmark_sums = defaultdict(float)
        
        for entry in history_data:
            for benchmark, score in entry['scores'].items():
                if score != 'N/A' and isinstance(score, (int, float)):
                    benchmark_sums[benchmark] += score
                    benchmark_counts[benchmark] += 1
        
        for benchmark in benchmark_list:
            if benchmark_counts[benchmark] > 0:
                benchmark_averages[benchmark] = benchmark_sums[benchmark] / benchmark_counts[benchmark]
            else:
                benchmark_averages[benchmark] = 'N/A'
        
        # Calculate summary statistics
        all_scores = []
        benchmarks_tested = set()
        
        for entry in history_data:
            for benchmark, score in entry['scores'].items():
                if score != 'N/A' and isinstance(score, (int, float)):
                    all_scores.append(score)
                    benchmarks_tested.add(benchmark)
        
        benchmark_stats = {
            'benchmarks_tested': len(benchmarks_tested),
            'overall_average': sum(all_scores) / len(all_scores) if all_scores else 0,
            'best_score': max(all_scores) if all_scores else 0
        }
        
        return render_template('llm/history.html',
                             model_name=model_name,
                             category=category,
                             history_data=history_data,
                             benchmark_list=benchmark_list,
                             benchmark_averages=benchmark_averages,
                             benchmark_stats=benchmark_stats)
    
    except Exception as e:
        print(f"Error loading history: {e}")
        return render_template('llm/history.html',
                             model_name=model_name,
                             category=category,
                             history_data=[],
                             benchmark_list=[],
                             benchmark_averages={},
                             benchmark_stats={'benchmarks_tested': 0, 'overall_average': 0, 'best_score': 0})



@app.route('/results/<model_name>')
def analyze(model_name):
    """Simplified results page - only current run."""
    if model_name not in MODELS:
        return f"Model '{model_name}' not found.", 404
    
    # Get current results only
    results_data = current_results.get(model_name, [])
    
    return render_template('results.html', model_name=model_name, results=results_data)


UPLOAD_FOLDER = 'uploads'
model_base_path = "models"
processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress
custom_evaluation_results = {}
processing_status = {}
custom_evaluation_progress = {} 
model_base_path = "models"
@app.route('/custom_llm/<model_name>')
def custom_llm(model_name):
    if model_name not in MODELS:
        flash(f"Unknown model: {model_name}")
        return redirect(url_for('index'))

    model_upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
    uploaded_files = []
    if os.path.exists(model_upload_dir):
        uploaded_files = [f for f in os.listdir(model_upload_dir) if os.path.isfile(os.path.join(model_upload_dir, f))]
    
    # Get current evaluation results only
    results = current_results.get(f"{model_name}_custom", {})
    
    return render_template('custom_llm.html', 
                         model_name=model_name, 
                         uploaded_files=uploaded_files,
                         evaluation_results=results)


@app.route('/run_custom_evaluation/<model_name>', methods=['POST'])
def run_custom_evaluation_route(model_name):
    if model_name not in MODELS:
        return jsonify({'error': 'Unknown model'}), 400
    
    try:
        # Import custom evaluator
        from llm_custom_utils import run_custom_evaluation
        
        # Get model path
        model_path = MODELS[model_name]
        upload_dir = os.path.join(UPLOAD_FOLDER)
        
        if not os.path.exists(upload_dir):
            return jsonify({'error': 'No files uploaded for evaluation'}), 400
        
        # Set processing status
        processing_status[f"{model_name}_custom"] = "processing"
        
        def background_evaluation():
            try:
                print(f"Starting custom evaluation for {model_name}")
                # Run custom evaluation
                results = run_custom_evaluation(model_name, model_path, upload_dir)
                print(f"Custom evaluation completed for {model_name}")
                
                # Store in current results only
                current_results[f"{model_name}_custom"] = results
                processing_status[f"{model_name}_custom"] = "complete"
                
            except Exception as e:
                print(f"Custom evaluation error for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                
                processing_status[f"{model_name}_custom"] = "error"
                current_results[f"{model_name}_custom"] = {"error": str(e)}
        
        # Run in background
        threading.Thread(target=background_evaluation, daemon=True).start()
        
        return jsonify({'status': 'started', 'message': 'Evaluation started successfully'})
        
    except Exception as e:
        print(f"Error starting evaluation: {e}")
        processing_status[f"{model_name}_custom"] = "error"
        return jsonify({'error': f'Error starting evaluation: {str(e)}'}), 500


@app.route('/clear_custom_results/<model_name>', methods=['POST'])
def clear_custom_results(model_name):
    """Clear custom evaluation results for a model."""
    try:
        # Clear from current results only
        custom_key = f"{model_name}_custom"
        if custom_key in current_results:
            del current_results[custom_key]
        
        # Clear processing status
        if custom_key in processing_status:
            del processing_status[custom_key]
        
        return jsonify({'status': 'success', 'message': 'Results cleared successfully'})
        
    except Exception as e:
        print(f"Error clearing results for {model_name}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500



@app.route('/check_custom_status/<model_name>')
def check_custom_status(model_name):
    status_key = f"{model_name}_custom"
    status = processing_status.get(status_key, "not_started")
    results = current_results.get(status_key, {})
    
    # Get progress information
    try:
        from llm_custom_utils import get_progress
        progress_info = get_progress(model_name)
    except:
        progress_info = {'stage': 0, 'message': 'Not started'}
    
    response_data = {
        "status": status,
        "results": results,
        "progress": progress_info,
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response_data)

@app.route('/download_custom_excel/<model_name>')
def download_custom_excel(model_name):
    """Download custom evaluation results as Excel file."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:
        # Get current evaluation results only
        results = current_results.get(f"{model_name}_custom", {})
        
        if not results or results.get('error'):
            flash("No evaluation results found for this model.")
            return redirect(url_for('custom_llm', model_name=model_name))
        
        # Same Excel generation logic as before...
        import pandas as pd
        from io import BytesIO
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main results sheet
            if 'ground_truth_comparison' in results:
                comparison_data = results['ground_truth_comparison']
                
                df_results = pd.DataFrame([
                    {
                        'Prompt': item['prompt'],
                        'Ground_Truth_Actual': item['actual'],
                        'Model_Extracted': item['extracted'],
                        'Confidence_Score': item['score'],
                        'Test_Grade': item['grade'],
                        'Status': 'Pass' if item['grade'] == '✅ Pass' else 'Fail' if item['grade'] == '❌ Fail' else 'Intermittent'
                    }
                    for item in comparison_data
                ])
                
                df_results.to_excel(writer, sheet_name='Evaluation_Results', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Model Name', 'Evaluation Date', 'Total Tests', 'Tests Passed',
                    'Tests Failed', 'Intermittent Tests', 'Overall Score (%)',
                    'Success Rate (%)', 'Average Score', 'Highest Score',
                    'Lowest Score', 'Files Processed'
                ],
                'Value': [
                    results.get('model_name', model_name),
                    results.get('timestamp', 'N/A'),
                    results.get('total_tests', 0),
                    results.get('pass_count', 0),
                    results.get('fail_count', 0),
                    results.get('intermittent_count', 0),
                    round(results.get('overall_score', 0), 2),
                    round(results.get('success_rate', 0), 2),
                    round(results.get('average_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('highest_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('lowest_score', 0), 2),
                    results.get('files_processed', 0)
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_custom_evaluation_results.xlsx"'
        
        return response
        
    except Exception as e:
        print(f"Error generating Excel file: {e}")
        flash(f"Error generating Excel file: {str(e)}")
        return redirect(url_for('custom_llm', model_name=model_name))



if __name__ == '__main__':
    app.run(debug=True)