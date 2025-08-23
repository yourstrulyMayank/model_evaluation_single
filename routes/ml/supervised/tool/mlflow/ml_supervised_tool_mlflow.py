from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory
from .ml_supervised_tool_mlflow_utils import (run_ml_evaluation_wrapper, 
 get_ml_progress, 
 convert_numpy_types, 
 get_ml_results,  
 export_results_to_json,
    clear_ml_progress,
    update_progress,    
    run_ml_evaluation,
    list_available_results,
    generate_ml_report,    
 )
import pandas as pd
import os

ml_s_t_mlflow_bp = Blueprint('ml_s_t_mlflow', __name__)

@ml_s_t_mlflow_bp.route('/evaluate_ml/<model_name>/<subcategory>', methods=['POST'])
def evaluate_ml(model_name, subcategory):
    """Evaluate ML models with subcategory support."""
    
    
    print(f"Evaluating ML model {model_name} on benchmark: MLFlow")   
    
    try:        
        # Correct path structure for your model
        model_path = os.path.join('models', model_name, 'model')
        dataset_path = os.path.join('models', model_name, 'dataset')        
        test_csv_path = os.path.join(dataset_path, 'test.csv')  # Look for test.csv in model directory
        model_file_path = os.path.join(model_path, 'model.pkl')  # Look for model.pkl
                
        
        # Validate paths exist
        if not os.path.exists(model_path):
            flash(f"Model directory not found: {model_path}")
            return redirect(url_for('index'))
        
        
        if not os.path.exists(model_file_path):
            flash(f"Model file not found: {model_file_path}")
            return redirect(url_for('index'))
        else:
            print(f'Found model: {model_file_path}')

        
        if not os.path.exists(test_csv_path):
            flash(f"Test CSV file not found: {test_csv_path}")
            return redirect(url_for('index'))
        else:
            print(f'Found test csv: {test_csv_path}')

        from concurrent.futures import ThreadPoolExecutor
        import threading
        evaluation_lock = threading.Lock()
        # Start evaluation in background thread
        # Start evaluation with proper resource management
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_ml_evaluation_wrapper,
                model_name, model_file_path, test_csv_path, 'MLFlow'
            )
        
        flash(f"ML model evaluation started for {model_name}. Check progress on the evaluation page.")
        
        return render_template('tool_evaluate_ml.html', 
                             model_name=model_name, 
                             subcategory=subcategory,
                             benchmark='MLFlow')
        
    except Exception as e:
        print(f"Error starting ML evaluation: {e}")
        flash(f"Error starting evaluation: {str(e)}")
        return redirect(url_for('index'))
    
# API endpoints for progress tracking and results
@ml_s_t_mlflow_bp.route('/api/ml_progress/<model_name>')
def get_ml_evaluation_progress(model_name):
    """Get current progress of ML model evaluation."""
    progress = get_ml_progress(model_name)
    return jsonify(progress)

@ml_s_t_mlflow_bp.route('/api/ml_progress/<model_name>')
def get_progress(model_name):
    """API endpoint to get evaluation progress"""
    progress = get_ml_progress(model_name)
    return jsonify(progress)

@ml_s_t_mlflow_bp.route('/api/ml_results/<model_name>')
def get_ml_evaluation_results(model_name):
    """Get evaluation results for a specific model."""
    results = get_ml_results(model_name)
    results = convert_numpy_types(results)
    return jsonify(results)

@ml_s_t_mlflow_bp.route('/api/export_results/<model_name>/<format>')
def export_results_api(model_name, format):
    """API endpoint to export results"""
    try:
        results = get_ml_results(model_name)
        if not results:
            return jsonify({"error": "No results found"}), 404
        
        if format == 'json':
            output_path = export_results_to_json(model_name)
            return send_file(output_path, as_attachment=True)
        elif format == 'csv':
            # Create CSV export
            output_dir = f"results/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"ml_evaluation_{model_name}.csv")
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_csv(csv_path, index=False)
            return send_file(csv_path, as_attachment=True)
        else:
            return jsonify({"error": "Invalid format"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ml_s_t_mlflow_bp.route('/api/start_evaluation/<model_name>', methods=['POST'])
def start_evaluation_api(model_name):
    """API endpoint to start model evaluation with proper resource management."""
    print('------------------------------------')
    print(f'Starting the evaluation api for model: {model_name}')
    try:
        # Get model and dataset paths from request or session
        model_path = request.json.get('model_path')
        dataset_path = request.json.get('dataset_path')
        print(f"Model path: {model_path}, Dataset path: {dataset_path}")
        
        if not model_path or not dataset_path:
            return jsonify({"error": "Model path and dataset path required"}), 400
        
        # Use ThreadPoolExecutor for better resource management
        from concurrent.futures import ThreadPoolExecutor
        
        def run_evaluation_task():
            try:
                run_ml_evaluation(model_name, model_path, dataset_path)
            except Exception as e:
                print(f"Evaluation task failed: {e}")
                update_progress(model_name, f"Error: {str(e)}", 0)
        
        # Submit task to thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(run_evaluation_task)
        
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_s_t_mlflow_bp.route('/api/start_evaluation/<model_name>', methods=['POST'])
def api_start_evaluation(model_name):
    """
    API endpoint to start or restart ML evaluation for a model.
    This will clear previous results and start a new evaluation.
    """
    try:
        # You may want to get subcategory and benchmark from request if needed
        subcategory = request.json.get('subcategory', 'supervised')
        benchmark =  'MLFlow'

        # Paths for model and test data
        
        model_path = os.path.join('models', 'ml', subcategory, model_name, 'model')
        dataset_path = os.path.join('models', 'ml', subcategory, model_name, 'dataset')        
        test_csv_path = os.path.join(dataset_path, 'test.csv')  # Look for test.csv in model directory
        model_file_path = os.path.join(model_path, 'model.pkl')  # Look for model.pkl
        # Validate paths
        if not os.path.exists(model_path) or not os.path.exists(test_csv_path):
            return jsonify({'error': 'Model or test data not found'}), 400

        # Clear previous progress/results
        clear_ml_progress(model_name)

        # Start evaluation in background
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(run_ml_evaluation, model_name, model_file_path, test_csv_path)

        return jsonify({'status': 'started', 'message': 'Evaluation started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@ml_s_t_mlflow_bp.route('/api/clear_ml_evaluation/<model_name>', methods=['POST'])
def clear_ml_evaluation_data(model_name):
    """Clear evaluation progress and results for a specific model."""
    try:
        clear_ml_progress(model_name)
        return jsonify({'status': 'success', 'message': f'Cleared evaluation data for {model_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@ml_s_t_mlflow_bp.route('/api/available_models')
def get_available_models():
    """API endpoint to get list of available models"""
    models = list_available_results()
    return jsonify(models)

@ml_s_t_mlflow_bp.route('/api/download_report/<model_name>')
def download_ml_report(model_name):
    """Download comprehensive evaluation report."""
    try:
        results = get_ml_results(model_name)
        if not results or 'error' in results:
            flash("No results available for download")
            return redirect(url_for('index'))
        
        # Generate report content
        report_content = generate_ml_report(results)
        
        # Create response
        response = make_response(report_content)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=ml_evaluation_report_{model_name}.json'
        
        return response
        
    except Exception as e:
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('index'))

