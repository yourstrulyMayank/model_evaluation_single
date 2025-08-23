from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory
import pandas as pd
import os
import traceback
import threading
from datetime import datetime
from .ml_supervised_custom_utils import (
    run_custom_ml_evaluation_task,
    clear_custom_ml_results,
    get_custom_ml_status,
    export_custom_ml_excel,
    export_custom_ml_csv,
    custom_evaluation_results,           
    custom_evaluation_progress,
    convert_numpy_types          
)
from werkzeug.utils import secure_filename
ml_s_c_bp = Blueprint('ml_s_c', __name__)

UPLOAD_FOLDER = 'uploads'
processing_status = {}  # Track per-model status
evaluation_progress = {}


@ml_s_c_bp.route('/custom_ml/<model_name>/<subcategory>')
def custom_ml(model_name, subcategory):
    """Custom ML evaluation page with results display"""
    try:
        # Get uploaded files
        upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
        uploaded_files = []
        if os.path.exists(upload_dir):
            uploaded_files = os.listdir(upload_dir)
        
        # Get evaluation results - THIS IS THE KEY FIX
        evaluation_results = None
        results_key = f"{model_name}_ml"
        if results_key in custom_evaluation_results:
            evaluation_results = custom_evaluation_results[results_key]
            print(f"Found evaluation results for {model_name}: {evaluation_results.keys()}")
        
        return render_template('custom_evaluate_ml.html', 
                             model_name=model_name,
                             subcategory=subcategory,
                             uploaded_files=uploaded_files,
                             evaluation_results=evaluation_results)  # Pass results here
    
    except Exception as e:
        print(f"Error in custom_ml route: {str(e)}")
        flash(f"Error loading page: {str(e)}")
        return redirect(url_for('index'))
    
@ml_s_c_bp.route('/run_custom_ml_evaluation/<model_name>', methods=['POST', 'GET'])
def run_custom_ml_evaluation(model_name):
    """Run custom ML evaluation with uploaded files and optional steps file."""
    try:
        upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
        if not os.path.exists(upload_dir):
            return jsonify({'error': 'No files uploaded for evaluation'}), 400
        
        # Find model, test, and steps files
        files = os.listdir(upload_dir)
        model_file = None
        test_file = None
        steps_file = None
        print(files)
        for file in files:
            if file.endswith(('.pkl', '.joblib', '.model')):
                model_file = os.path.join(upload_dir, file)
            elif file.endswith(('.xlsx', '.xls', '.csv')):
                test_file = os.path.join(upload_dir, file)
            elif file.startswith('steps.') and file.split('.')[-1] in ('py', 'json', 'txt'):
                steps_file = os.path.join(upload_dir, file)
        
        if not model_file or not test_file:
            return jsonify({'error': 'Both model file (.pkl) and test file (.xlsx/.csv) are required'}), 400
        
        # Set initial status
        processing_status[f"{model_name}_ml_custom"] = "processing"
        
        def background_evaluation():
            """Background evaluation with comprehensive error handling"""
            try:
                print(f"Starting ML evaluation for model: {model_name}")
                print(f"Model file: {model_file}")
                print(f"Test file: {test_file}")
                print(f"Steps file: {steps_file}")
                
                # Run the evaluation
                result = run_custom_ml_evaluation_task(model_name, model_file, test_file, steps_file)
                
                print(f"Evaluation completed for {model_name}")
                if result.get('error'):
                    print(f"Evaluation error: {result['error']}")
                    processing_status[f"{model_name}_ml_custom"] = "error"
                else:
                    print(f"Evaluation successful for {model_name}")
                    processing_status[f"{model_name}_ml_custom"] = "complete"
                    
            except Exception as e:
                error_msg = f"Background evaluation failed: {str(e)}"
                print(f"{error_msg}\n{traceback.format_exc()}")
                
                # Store error in results
                custom_evaluation_results[f"{model_name}_ml"] = {
                    'error': error_msg,
                    'traceback': traceback.format_exc(),
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
                processing_status[f"{model_name}_ml_custom"] = "error"
        
        # Start background thread - removed daemon=True to prevent immediate shutdown
        thread = threading.Thread(target=background_evaluation)
        thread.start()
        
        print(f"Background thread started for {model_name}")
        return jsonify({'status': 'started', 'message': 'ML evaluation started successfully'})
        
    except Exception as e:
        error_msg = f'Error starting evaluation: {str(e)}'
        print(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500


@ml_s_c_bp.route('/check_custom_ml_status/<model_name>')
def check_custom_ml_status(model_name):
    """Check status of custom ML evaluation - FIXED VERSION"""
    try:
        status = processing_status.get(f"{model_name}_ml_custom", "not_started")
        results = custom_evaluation_results.get(f"{model_name}_ml", {})
        progress = custom_evaluation_progress.get(model_name, {})

        
        
        if status == "complete" and results and not results.get('error'):
            return jsonify(convert_numpy_types({
                'status': 'complete',
                'results': results,
                'progress': progress
            }))
        elif status == "error" or results.get('error'):
            return jsonify(convert_numpy_types({
                'status': 'error',
                'results': results,
                'progress': progress
            }))
        else:
            return jsonify(convert_numpy_types({
                'status': 'processing',
                'progress': progress
            }))
            
    except Exception as e:
        print(f"Error checking status for {model_name}: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500



@ml_s_c_bp.route('/download_custom_ml_report/<model_name>')
def download_custom_ml_report(model_name):
    """Download custom ML evaluation report as Excel."""
    try:
        excel_path = os.path.join('uploads', model_name, f"{model_name}_custom_ml_report.xlsx")
        print(excel_path)
        if not os.path.exists(excel_path):
            print("file not found")
            flash("Excel report not found. Please re-run evaluation.")
            return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))
        return send_file(excel_path, as_attachment=True, download_name=f"{model_name}_custom_ml_report.xlsx")
    except Exception as e:
        print(f"Error downloading report for {model_name}: {str(e)}")
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))



@ml_s_c_bp.route('/download_custom_ml_testcases/<model_name>')
def download_custom_ml_testcases(model_name):
    """Download test cases with predictions as CSV."""
    try:
        csv_path = os.path.join('uploads', model_name, f"{model_name}_test_results.csv")
        print(csv_path)
        if not os.path.exists(csv_path):
            print("file not found")
            flash("CSV results not found. Please re-run evaluation.")
            return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))
        return send_file(csv_path, as_attachment=True, download_name=f"{model_name}_test_results.csv")
    except Exception as e:
        print(f"Error downloading test cases for {model_name}: {str(e)}")
        flash(f"Error generating CSV: {str(e)}")
        return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))


@ml_s_c_bp.route('/clear_custom_ml_results/<model_name>', methods=['POST'])
def clear_custom_ml_results_route(model_name):
    """Clear custom evaluation results for a model."""
    try:
        clear_custom_ml_results(model_name)
        print(f"Results cleared for {model_name}")
        return jsonify({'status': 'success', 'message': 'Results cleared successfully'})
    except Exception as e:
        print(f"Error clearing results for {model_name}: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ml_s_c_bp.route('/upload_ml_files/<model_name>', methods=['POST'])
def upload_ml_files(model_name):
    """Upload model and test files for custom ML evaluation."""
    try:
        model_upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
        os.makedirs(model_upload_dir, exist_ok=True)
        
        uploaded_files = []
        
        # Handle model file upload
        if 'model_file' in request.files:
            model_file = request.files['model_file']
            if model_file.filename != '':
                filename = secure_filename(model_file.filename)
                model_path = os.path.join(model_upload_dir, filename)
                model_file.save(model_path)
                uploaded_files.append(f"Model: {filename}")
        
        # Handle test data upload
        if 'test_file' in request.files:
            test_file = request.files['test_file']
            if test_file.filename != '':
                filename = secure_filename(test_file.filename)
                test_path = os.path.join(model_upload_dir, filename)
                test_file.save(test_path)
                uploaded_files.append(f"Test Data: {filename}")
        
        if uploaded_files:
            return jsonify({
                'status': 'success', 
                'message': f'Uploaded: {", ".join(uploaded_files)}'
            })
        else:
            return jsonify({'status': 'error', 'message': 'No files were uploaded'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
