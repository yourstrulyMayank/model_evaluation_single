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
        # Get current results only
        results_data = current_results.get(model_name, [])
        
        if not results_data:
            flash("No evaluation results found for this model.")
            return redirect(url_for('analyze', model_name=model_name))
        
        # Create a comprehensive HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Evaluation Report - {model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ text-align: center; margin-bottom: 40px; border-bottom: 2px solid #333; padding-bottom: 20px; }}
                .summary {{ margin-bottom: 30px; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .summary-card {{ border: 1px solid #ddd; padding: 15px; text-align: center; background: #f9f9f9; }}
                .score {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .task-section {{ margin-bottom: 30px; page-break-inside: avoid; }}
                .task-header {{ background: #f5f5f5; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 10px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0; }}
                .metric-item {{ text-align: center; padding: 10px; background: #f9f9f9; border: 1px solid #ddd; }}
                .samples-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .samples-table th, .samples-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .samples-table th {{ background-color: #f5f5f5; }}
                .code {{ background: #f8f8f8; padding: 5px; font-family: monospace; white-space: pre-wrap; }}
                @media print {{ body {{ margin: 20px; }} .page-break {{ page-break-before: always; }} }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BIG-bench Evaluation Report</h1>
                <h2>{model_name}</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        if results_data:
            latest = results_data[0]
            
            # Overall Summary
            if 'summary' in latest and latest['summary']:
                html_content += """
                <div class="summary">
                    <h2>Performance Summary</h2>
                    <div class="summary-grid">
                """
                
                if 'overall' in latest['summary']:
                    overall = latest['summary']['overall']
                    html_content += f"""
                    <div class="summary-card">
                        <h3>Overall Score</h3>
                        <div class="score">{overall['mean']*100:.1f}%</div>
                        <p>± {overall['std']*100:.1f}% ({overall['count']} tasks)</p>
                    </div>
                    """
                
                for task_type, stats in latest['summary'].items():
                    if task_type != 'overall':
                        html_content += f"""
                        <div class="summary-card">
                            <h3>{task_type.title()}</h3>
                            <div class="score">{stats['mean']*100:.1f}%</div>
                            <p>± {stats['std']*100:.1f}% ({stats['count']} tasks)</p>
                        </div>
                        """
                
                html_content += "</div></div>"
            
            # Detailed Results
            if 'detailed_results' in latest:
                html_content += "<h2>Detailed Task Results</h2>"
                
                for i, task_result in enumerate(latest['detailed_results']):
                    if i > 0:  # Add page break between tasks for better printing
                        html_content += '<div class="page-break"></div>'
                    
                    primary_score = task_result.get('summary', {}).get('primary_metric', {}).get('mean', 0)
                    html_content += f"""
                    <div class="task-section">
                        <div class="task-header">
                            <h3>{task_result['task']}</h3>
                            <p>Type: {task_result.get('task_type', 'N/A')} | Primary Score: {primary_score*100:.1f}%</p>
                        </div>
                    """
                    
                    # Metrics
                    if 'summary' in task_result:
                        html_content += '<div class="metrics-grid">'
                        for metric_name, metric_data in task_result['summary'].items():
                            html_content += f"""
                            <div class="metric-item">
                                <strong>{metric_name.replace('_', ' ').title()}</strong><br>
                                {metric_data['mean']:.3f}
                            </div>
                            """
                        html_content += '</div>'
                    
                    # Sample results
                    if 'samples' in task_result and task_result['samples']:
                        html_content += """
                        <h4>Sample Predictions</h4>
                        <table class="samples-table">
                            <tr>
                                <th>Example</th>
                                <th>Input</th>
                                <th>Generated</th>
                                <th>Expected</th>
                                <th>Score</th>
                            </tr>
                        """
                        
                        for sample in task_result['samples'][:3]:  # Show first 3 samples
                            score = sample.get('metrics', {}).get('primary_metric', 0)
                            html_content += f"""
                            <tr>
                                <td>{sample.get('example_number', 'N/A')}</td>
                                <td><div class="code">{sample.get('input', '')[:200]}{'...' if len(str(sample.get('input', ''))) > 200 else ''}</div></td>
                                <td><div class="code">{sample.get('generated', '')}</div></td>
                                <td><div class="code">{sample.get('expected', '')}</div></td>
                                <td>{score:.2f}</td>
                            </tr>
                            """
                        
                        html_content += '</table>'
                    
                    html_content += '</div>'
        
        html_content += f"""
            <div class="summary" style="margin-top: 40px; border-top: 1px solid #ddd; padding-top: 20px;">
                <p><strong>Report generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total tasks evaluated:</strong> {len(latest.get('detailed_results', []))}</p>
            </div>
        </body>
        </html>
        """
        
        # Try to generate PDF if weasyprint is available
        try:
            from weasyprint import HTML
            pdf_buffer = BytesIO()
            HTML(string=html_content).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)
            
            response = make_response(pdf_buffer.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_evaluation_report.pdf"'
            return response
            
        except ImportError:
            # Fallback: return HTML if weasyprint not available
            response = make_response(html_content)
            response.headers['Content-Type'] = 'text/html'
            response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_evaluation_report.html"'
            return response
        
    except Exception as e:
        print(f"Error generating report: {e}")
        flash(f"Error generating report: {str(e)}")
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
    """Enhanced status check with strict completion validation."""
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
    
    # Enhanced completion validation
    progress_stage = progress.get('stage', 0)
    has_results = model_name in current_results and len(current_results[model_name]) > 0
    
    # Only mark as complete if BOTH conditions are met:
    # 1. Progress stage indicates completion (>= 6)
    # 2. Results are actually stored
    if progress_stage >= 6 and has_results and status != "complete":
        status = "complete" 
        processing_status[model_name] = "complete"
        print(f"✅ Marked {model_name} as complete")
    elif progress_stage == -1:
        status = "error"
        processing_status[model_name] = "error"
    elif progress_stage > 0 and progress_stage < 6:
        status = "processing"
        processing_status[model_name] = "processing"
    
    # Additional validation: if status is "complete" but no results, reset to processing
    if status == "complete" and not has_results:
        print(f"⚠️ Status was 'complete' but no results found for {model_name}, reverting to processing")
        status = "processing"
        processing_status[model_name] = "processing"
    
    # Debug logging - REMOVE DUPLICATE
    print(f"📊 Status check for {model_name}: status={status}, progress_stage={progress_stage}, has_results={has_results}")
    
    return jsonify({
        "status": status,
        "progress": progress,
        "has_results": has_results,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/results/<model_name>')
def analyze(model_name):
    """Results page with strict completion validation."""
    if model_name not in MODELS:
        return f"Model '{model_name}' not found.", 404
    
    # Check if evaluation is actually complete
    status = processing_status.get(model_name, "not_started")
    results_data = current_results.get(model_name, [])
    
    # If evaluation is not complete or no results, redirect to loading/index
    if status != "complete" or not results_data:
        if status == "processing":
            print(f"📊 Evaluation still in progress for {model_name}, redirecting to loading page")
            return redirect(url_for('evaluate_llm', model_name=model_name))
        elif status == "error":
            flash(f"Evaluation failed for {model_name}. Please try again.")
            return redirect(url_for('index'))
        else:
            flash(f"No evaluation results found for {model_name}. Please run evaluation first.")
            return redirect(url_for('index'))
    
    print(f"📊 Displaying results for {model_name}: {len(results_data)} result sets")
    return render_template('results.html', model_name=model_name, history=results_data)


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