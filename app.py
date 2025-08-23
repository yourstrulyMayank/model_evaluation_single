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
# from weasyprint import HTML, CSS

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
    """Generate PDF report with charts and tables from results page."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:
        # Load results from file
        from llm_tool_bigbench_utils import load_results_from_file
        results_data = load_results_from_file(model_name)
        
        if not results_data:
            flash("No evaluation results found for this model.")
            return redirect(url_for('analyze', model_name=model_name))
        
        latest = results_data[0]
        
        # Create comprehensive HTML with charts and styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Evaluation Report - {model_name}</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #007bff; padding-bottom: 20px; }}
                .header h1 {{ color: #007bff; margin-bottom: 5px; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .summary-card {{ border: 1px solid #ddd; padding: 15px; text-align: center; background: #f8f9fa; border-radius: 8px; }}
                .summary-card h3 {{ margin-top: 0; color: #495057; font-size: 1.1em; }}
                .score {{ font-size: 2.2em; font-weight: bold; color: #007bff; margin: 10px 0; }}
                .details {{ color: #6c757d; font-size: 0.9em; }}
                .chart-container {{ margin: 30px 0; text-align: center; page-break-inside: avoid; }}
                .chart-container h3 {{ color: #495057; margin-bottom: 15px; }}
                .chart-placeholder {{ width: 100%; height: 300px; background: #f8f9fa; border: 1px solid #ddd; display: flex; align-items: center; justify-content: center; color: #6c757d; }}
                .task-section {{ margin-bottom: 25px; page-break-inside: avoid; }}
                .task-header {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 10px; border-radius: 0 8px 8px 0; }}
                .task-header h3 {{ margin: 0; color: #495057; }}
                .task-type {{ color: #6c757d; font-size: 0.9em; font-weight: normal; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0; }}
                .metric-item {{ text-align: center; padding: 10px; background: #fff; border: 1px solid #dee2e6; border-radius: 5px; }}
                .metric-item strong {{ color: #495057; font-size: 0.9em; }}
                .samples-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
                .samples-table th {{ background-color: #007bff; color: white; padding: 8px; text-align: left; }}
                .samples-table td {{ border: 1px solid #dee2e6; padding: 8px; vertical-align: top; }}
                .samples-table pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; font-size: 0.8em; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; margin-top: 20px; text-align: center; }}
                @media print {{ 
                    body {{ margin: 15px; }} 
                    .page-break {{ page-break-before: always; }}
                    .chart-placeholder {{ height: 200px; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BIG-bench Evaluation Report</h1>
                <h2>{model_name}</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        if 'summary' in latest and latest['summary']:
            # Overall Summary Cards
            html_content += """
            <div class="summary-grid">
            """
            
            if 'overall' in latest['summary']:
                overall = latest['summary']['overall']
                html_content += f"""
                <div class="summary-card">
                    <h3>Overall Performance</h3>
                    <div class="score">{overall['mean']*100:.1f}%</div>
                    <div class="details">± {overall['std']*100:.1f}% ({overall['count']} tasks)</div>
                </div>
                """
            
            for task_type, stats in latest['summary'].items():
                if task_type != 'overall':
                    html_content += f"""
                    <div class="summary-card">
                        <h3>{task_type.title()}</h3>
                        <div class="score">{stats['mean']*100:.1f}%</div>
                        <div class="details">± {stats['std']*100:.1f}% ({stats['count']} tasks)</div>
                    </div>
                    """
            
            html_content += "</div>"
            
            # Performance Chart Placeholder
            html_content += """
            <div class="chart-container">
                <h3>Performance by Task Type</h3>
                <div class="chart-placeholder">
                    <div>
                        <strong>Performance Summary</strong><br>
            """
            
            for task_type, stats in latest['summary'].items():
                if task_type != 'overall':
                    html_content += f"{task_type.title()}: {stats['mean']*100:.1f}%<br>"
            
            html_content += """
                    </div>
                </div>
            </div>
            """
        
        # Detailed Task Results
        if 'detailed_results' in latest:
            html_content += "<h2 style='color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 10px;'>Detailed Task Results</h2>"
            
            for i, task_result in enumerate(latest['detailed_results']):
                if i > 0:
                    html_content += '<div class="page-break"></div>'
                
                primary_score = task_result.get('summary', {}).get('primary_metric', {}).get('mean', 0)
                html_content += f"""
                <div class="task-section">
                    <div class="task-header">
                        <h3>{task_result['task']} <span class="task-type">({task_result.get('task_type', 'N/A')})</span></h3>
                        <p style="margin: 5px 0 0 0;">Primary Score: <strong>{primary_score*100:.1f}%</strong></p>
                    </div>
                """
                
                # Metrics Grid
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
                
                # Sample Results Table
                if 'samples' in task_result and task_result['samples']:
                    html_content += """
                    <h4 style="color: #495057;">Sample Predictions</h4>
                    <table class="samples-table">
                        <thead>
                            <tr>
                                <th style="width: 5%">#</th>
                                <th style="width: 35%">Input</th>
                                <th style="width: 25%">Generated</th>
                                <th style="width: 25%">Expected</th>
                                <th style="width: 10%">Score</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    
                    for sample in task_result['samples'][:3]:
                        score = sample.get('metrics', {}).get('primary_metric', 0)
                        input_text = str(sample.get('input', ''))[:150] + ('...' if len(str(sample.get('input', ''))) > 150 else '')
                        html_content += f"""
                        <tr>
                            <td>{sample.get('example_number', 'N/A')}</td>
                            <td><pre>{input_text}</pre></td>
                            <td><pre>{sample.get('generated', '')}</pre></td>
                            <td><pre>{sample.get('expected', '')}</pre></td>
                            <td style="text-align: center;"><strong>{score:.2f}</strong></td>
                        </tr>
                        """
                    
                    html_content += '</tbody></table>'
                
                html_content += '</div>'
        
        html_content += f"""
            <div class="timestamp">
                <strong>Report generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Total tasks evaluated:</strong> {len(latest.get('detailed_results', []))}
            </div>
        </body>
        </html>
        """
        
        # Try to generate PDF
        try:
            pdf_buffer = BytesIO()
            HTML(string=html_content).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)
            
            response = make_response(pdf_buffer.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename="{model_name.replace(" ", "_")}_evaluation_report.pdf"'
            return response
            
        except ImportError:
            # Fallback: return HTML
            response = make_response(html_content)
            response.headers['Content-Type'] = 'text/html'
            response.headers['Content-Disposition'] = f'attachment; filename="{model_name.replace(" ", "_")}_evaluation_report.html"'
            return response
        
    except Exception as e:
        print(f"Error generating report: {e}")
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('analyze', model_name=model_name))
        
@app.route('/export_json/<model_name>')
def export_json(model_name):
    """Export evaluation_results.json file directly."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:
        from llm_tool_bigbench_utils import RESULTS_FILE
        
        # Check if results file exists
        if not os.path.exists(RESULTS_FILE):
            flash("No evaluation results file found.")
            return redirect(url_for('analyze', model_name=model_name))
        
        # Send the file directly
        return send_file(
            RESULTS_FILE,
            as_attachment=True,
            download_name=f"evaluation_results_{model_name.replace(' ', '_')}.json",
            mimetype='application/json'
        )
        
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

    # Clear previous results
    try:
        from llm_tool_bigbench_utils import clear_results_file
        clear_results_file()
    except Exception as e:
        print(f"⚠️ Warning: Could not clear results file: {e}")

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
    
    # Check if results exist in file
    try:
        from llm_tool_bigbench_utils import load_results_from_file
        file_results = load_results_from_file(model_name)
        has_results = len(file_results) > 0
    except:
        has_results = False
    
    # Only mark as complete if BOTH conditions are met:
    if progress_stage >= 5 and has_results and status != "complete":
        status = "complete" 
        processing_status[model_name] = "complete"
        print(f"✅ Marked {model_name} as complete")
    elif progress_stage == -1:
        status = "error"
        processing_status[model_name] = "error"
    elif progress_stage > 0 and progress_stage < 5:
        status = "processing"
        processing_status[model_name] = "processing"
    
    # Debug logging - add this right before the return statement
    print(f"📊 Status check for {model_name}: status={status}, progress_stage={progress_stage}, has_results={has_results}")
    print(f"📊 Available models in current_results: {list(current_results.keys())}")
    if model_name in current_results:
        print(f"📊 Results for {model_name}: {len(current_results[model_name])} entries")
    
    return jsonify({
        "status": status,
        "progress": progress,
        "has_results": has_results,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/clear_results/<model_name>')
def clear_model_results(model_name):
    """Clear results for a specific model before starting evaluation."""
    try:
        from llm_tool_bigbench_utils import clear_results_file
        clear_results_file()
        print(f"🧹 Cleared results file before evaluating {model_name}")
        return jsonify({'status': 'cleared'})
    except Exception as e:
        print(f"❌ Error clearing results: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/results/<model_name>')
def analyze(model_name):
    """Results page loading from file."""
    if model_name not in MODELS:
        return f"Model '{model_name}' not found.", 404
    
    # Load results from file
    try:
        from llm_tool_bigbench_utils import load_results_from_file
        results_data = load_results_from_file(model_name)
    except Exception as e:
        print(f"❌ Error loading results from file: {e}")
        results_data = []
    
    # Check if evaluation is complete and has results
    status = processing_status.get(model_name, "not_started")
    
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

    # Get current evaluation results only
    results = current_results.get(f"{model_name}_custom", {})
    
    return render_template('custom_llm.html', 
                         model_name=model_name, 
                         evaluation_results=results)

@app.route('/run_custom_evaluation/<model_name>', methods=['POST'])
def run_custom_evaluation_route(model_name):
    if model_name not in MODELS:
        return jsonify({'error': 'Unknown model'}), 400
    
    try:
        # Import custom evaluator
        from custom_evaluate_llm import run_custom_evaluation
        
        # For custom_llm, get model path from custom_models folder
        # Determine model path based on evaluation type
        wealth_advisory_dir = os.path.join(upload_dir, 'wealth_advisory')
        compliance_dir = os.path.join(upload_dir, 'compliance')
        
        if os.path.exists(wealth_advisory_dir):
            model_path = f"custom_models/wealth_advisory/{model_name}"
        elif os.path.exists(compliance_dir):
            model_path = f"custom_models/compliance/{model_name}"  
        else:
            model_path = MODELS[model_name]  # Fallback to original MODELS dict
        upload_dir = os.path.join(UPLOAD_FOLDER)
        
        if not os.path.exists(upload_dir):
            return jsonify({'error': 'Upload directory not found'}), 400
        
        # Check if wealth_advisory or compliance folder exists
        wealth_advisory_dir = os.path.join(upload_dir, 'wealth_advisory')
        compliance_dir = os.path.join(upload_dir, 'compliance')
        
        if not os.path.exists(wealth_advisory_dir) and not os.path.exists(compliance_dir):
            return jsonify({'error': 'Neither wealth_advisory nor compliance folder found in uploads'}), 400
        
        # Set processing status
        processing_status[f"{model_name}_custom"] = "processing"
        
        def background_evaluation():
            try:
                print(f"Starting custom RAG evaluation with NLP metrics for {model_name}")
                # Run custom evaluation with NLP pipeline
                results = run_custom_evaluation(model_name, model_path, upload_dir)
                print(f"Custom evaluation with NLP metrics completed for {model_name}")
                
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
        
        return jsonify({'status': 'started', 'message': 'Evaluation with NLP metrics started successfully'})
        
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
        
        # Clear progress tracking
        from custom_evaluate_llm import clear_progress
        clear_progress(model_name)
        
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
        from custom_evaluate_llm import get_progress
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
    """Download custom evaluation results as Excel file with NLP metrics."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:
        # Get current evaluation results only
        results = current_results.get(f"{model_name}_custom", {})
        
        if not results or results.get('error'):
            flash("No evaluation results found for this model.")
            return redirect(url_for('custom_llm', model_name=model_name))
        
        import pandas as pd
        from io import BytesIO
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main results sheet (UI compatible format)
            if 'ground_truth_comparison' in results:
                comparison_data = results['ground_truth_comparison']
                
                df_results = pd.DataFrame([
                    {
                        'Prompt': item['prompt'],
                        'Ground_Truth_Actual': item['actual'],
                        'Model_Extracted': item['extracted'],
                        'Confidence_Score': item['score'],
                        'Test_Grade': item['grade'],
                        'Status': 'Pass' if item['grade'] == '✅ Pass' else 'Fail'
                    }
                    for item in comparison_data
                ])
                
                df_results.to_excel(writer, sheet_name='Evaluation_Results', index=False)
            
            # Detailed NLP metrics sheet
            if 'detailed_nlp_results' in results:
                nlp_df = pd.DataFrame(results['detailed_nlp_results'])
                # Remove columns that might not be needed for Excel export
                columns_to_exclude = ['GT_Entities', 'Pred_Entities']
                nlp_df = nlp_df.drop(columns=[col for col in columns_to_exclude if col in nlp_df.columns])
                nlp_df.to_excel(writer, sheet_name='Detailed_NLP_Metrics', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Model Name', 'Evaluation Type', 'Evaluation Date', 'Total Tests', 
                    'Tests Passed', 'Tests Failed', 'Overall Score (%)',
                    'Success Rate (%)', 'Average Score', 'Highest Score', 'Lowest Score',
                    'Median Score', 'Standard Deviation'
                ],
                'Value': [
                    results.get('model_name', model_name),
                    results.get('evaluation_type', 'N/A'),
                    results.get('timestamp', 'N/A'),
                    results.get('total_tests', 0),
                    results.get('pass_count', 0),
                    results.get('fail_count', 0),
                    round(results.get('overall_score', 0), 2),
                    round(results.get('success_rate', 0), 2),
                    round(results.get('average_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('highest_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('lowest_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('median_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('std_deviation', 0), 2)
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_custom_nlp_evaluation_results.xlsx"'
        
        return response
        
    except Exception as e:
        print(f"Error generating Excel file: {e}")
        flash(f"Error generating Excel file: {str(e)}")
        return redirect(url_for('custom_llm', model_name=model_name))

# PORT = 8056


## ML Imports #
from routes.ml.supervised.tool.mlflow.ml_supervised_tool_mlflow import ml_s_t_mlflow_bp
from routes.ml.supervised.custom.ml_supervised_custom import ml_s_c_bp
## ML Blueprints ##
app.register_blueprint(ml_s_t_mlflow_bp)
app.register_blueprint(ml_s_c_bp)


if __name__ == '__main__':
    app.run(debug=True)