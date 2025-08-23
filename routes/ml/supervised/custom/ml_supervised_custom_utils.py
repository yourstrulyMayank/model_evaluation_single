"""
Custom ML Model Evaluation Module
Handles ML model evaluation functionality separately from main Flask app
"""

import os
import pandas as pd
import numpy as np
import pickle
import joblib
import importlib.util
import logging
import traceback
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# In-memory storage for results and status (should be replaced with persistent storage in production)
custom_evaluation_results = {}
processing_status = {}
custom_evaluation_progress = {} 


class CustomMLEvaluator:
    """Custom ML Model Evaluator Class with improved error handling"""
    
    def __init__(self):
        self.current_stage = 0
        self.progress_messages = {
            1: "Loading and validating model file...",
            2: "Processing and preparing test data...",
            3: "Making predictions on test data...",
            4: "Calculating performance metrics...",
            5: "Finalizing evaluation results..."
        }
    
    def update_progress(self, stage, custom_message=None):
        """Update progress stage and message"""
        self.current_stage = stage
        message = custom_message or self.progress_messages.get(stage, "Processing...")
        print(f"Progress update: Stage {stage} - {message}")
        return {"stage": stage, "message": message}
    
    def load_model(self, model_file_path):
        """Load ML model from file with better error handling"""
        try:
            print(f"Loading model from: {model_file_path}")
            
            if not os.path.exists(model_file_path):
                return None, f"Model file not found: {model_file_path}"
            
            # Try pickle first
            try:
                with open(model_file_path, 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded successfully using pickle")
                return model, None
            except Exception as pickle_error:
                print(f"Pickle loading failed: {str(pickle_error)}")
                
                # Try joblib
                try:
                    model = joblib.load(model_file_path)
                    print("Model loaded successfully using joblib")
                    return model, None
                except Exception as joblib_error:
                    print(f"Joblib loading failed: {str(joblib_error)}")
                    raise Exception(f"Failed with both pickle ({str(pickle_error)}) and joblib ({str(joblib_error)})")
                    
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def load_test_data(self, test_file_path, steps_path=None):
        """Load test data from file with better error handling"""
        try:
            print(f"Loading test data from: {test_file_path}")
            if not os.path.exists(test_file_path):
                return None, None, f"Test file not found: {test_file_path}"

            # Load data based on file extension
            if test_file_path.endswith('.csv'):
                data = pd.read_csv(test_file_path)
            elif test_file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(test_file_path)
            else:
                return None, None, f"Unsupported file format: {test_file_path}"

            if data.empty:
                return None, None, "Test file is empty"

            print(f"Test data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

            # Apply preprocessing if steps_path is provided
            if steps_path:
                print(f"Applying preprocessing from: {steps_path}")
                # Expect preprocess to return (X, y)
                X_test, y_test = self.apply_preprocessing(data, steps_path)
            else:
                # Assume last column is target, rest are features
                X_test = data.iloc[:, :-1]
                y_test = data.iloc[:, -1]

            print(f"Final test data shape: X_test {X_test.shape}, y_test {y_test.shape}")
            return X_test, y_test, None

        except Exception as e:
            error_msg = f"Failed to load or preprocess test data: {str(e)}"
            print(error_msg)
            return None, None, error_msg
            
        
    def create_predictions_data(self, y_true, y_pred, problem_type, X_test=None):
        """Create structured predictions data for display with original test data"""
        try:
            predictions_list = []
            
            for i, (actual, predicted) in enumerate(zip(y_true, y_pred)):
                prediction_data = {
                    'test_case': i + 1,
                    'actual': str(actual),
                    'predicted': str(predicted)
                }
                
                # Add original test features if available
                if X_test is not None:
                    try:
                        # Get the row data for this test case
                        if hasattr(X_test, 'iloc'):
                            row_data = X_test.iloc[i]
                            for col_name, col_value in row_data.items():
                                prediction_data[f'feature_{col_name}'] = col_value
                        elif isinstance(X_test, np.ndarray):
                            for j, feature_value in enumerate(X_test[i]):
                                prediction_data[f'feature_{j}'] = feature_value
                    except Exception as e:
                        print(f"Error adding features for row {i}: {str(e)}")
                
                if problem_type == 'classification':
                    # For classification, show if prediction is correct (no emojis)
                    is_correct = str(actual) == str(predicted)
                    prediction_data['result'] = 'Correct' if is_correct else 'Incorrect'
                    prediction_data['difference'] = 'Match' if is_correct else 'Mismatch'
                else:
                    # For regression, show numerical difference
                    try:
                        actual_num = float(actual)
                        pred_num = float(predicted)
                        difference = abs(actual_num - pred_num)
                        prediction_data['result'] = f'Error: {difference:.4f}'
                        prediction_data['difference'] = f'{difference:.4f}'
                    except:
                        prediction_data['result'] = 'N/A'
                        prediction_data['difference'] = 'N/A'
                
                predictions_list.append(prediction_data)
            
            print(f"Created predictions data: {len(predictions_list)} predictions")
            return predictions_list
            
        except Exception as e:
            print(f"Error creating predictions data: {str(e)}")
            return []

    def apply_preprocessing(self, data, steps_path):
        """Apply preprocessing steps from a file"""
        try:
            print(f"Applying preprocessing from: {steps_path}")
            if not os.path.exists(steps_path):
                logger.warning(f"Preprocessing file not found: {steps_path}")
                return data
            if steps_path.endswith('.py'):
                return self._apply_python_preprocessing(data, steps_path)
            elif steps_path.endswith('.json'):
                return self._apply_json_preprocessing(data, steps_path)
            else:
                logger.warning(f"Unsupported preprocessing file type: {steps_path}")
                return data
        except Exception as e:
            print(f"Error applying preprocessing: {str(e)}")
            return data

    def _apply_python_preprocessing(self, data, steps_path):
        """Apply preprocessing from Python file"""
        try:
            spec = importlib.util.spec_from_file_location("preprocessing", steps_path)
            preprocessing_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocessing_module)
            if hasattr(preprocessing_module, 'preprocess'):
                result = preprocessing_module.preprocess(data)
                # If preprocess returns a tuple (X, y), return as is
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                # Otherwise, assume only X is returned, y is last column
                X = result
                y = data.iloc[:, -1]
                return X, y
            elif hasattr(preprocessing_module, 'transform'):
                X = preprocessing_module.transform(data)
                y = data.iloc[:, -1]
                return X, y
            else:
                logger.warning("No preprocess() or transform() function found in Python file")
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
                return X, y
        except Exception as e:
            print(f"Error applying Python preprocessing: {str(e)}")
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            return X, y

    def _apply_json_preprocessing(self, X_test, steps_path):
        """Apply preprocessing from JSON configuration"""
        try:
            import json
            
            with open(steps_path, 'r') as f:
                config = json.load(f)
            
            X_processed = X_test.copy()
            
            # Handle common preprocessing steps
            if 'fill_na' in config:
                fill_value = config['fill_na']
                X_processed = X_processed.fillna(fill_value)
                print(f"Applied fillna with value: {fill_value}")
            
            if 'drop_columns' in config:
                columns_to_drop = config['drop_columns']
                existing_columns = [col for col in columns_to_drop if col in X_processed.columns]
                if existing_columns:
                    X_processed = X_processed.drop(columns=existing_columns)
                    print(f"Dropped columns: {existing_columns}")
            
            if 'scale_columns' in config:
                # Simple min-max scaling
                columns_to_scale = config['scale_columns']
                for col in columns_to_scale:
                    if col in X_processed.columns:
                        col_min = X_processed[col].min()
                        col_max = X_processed[col].max()
                        if col_max != col_min:
                            X_processed[col] = (X_processed[col] - col_min) / (col_max - col_min)
                            print(f"Scaled column: {col}")
            
            return X_processed
            
        except Exception as e:
            print(f"Error applying JSON preprocessing: {str(e)}")
            return X_test
    
    # def run_evaluation(self, model_name, model_file_path, test_file_path, steps_path=None, progress_callback=None):
    #     """Run complete ML model evaluation with comprehensive error handling"""
    #     results = {
    #         'model_name': model_name,
    #         'timestamp': datetime.now().isoformat(),
    #         'files_processed': 0,
    #         'file_info': {
    #             'model_file': os.path.basename(model_file_path),
    #             'test_file': os.path.basename(test_file_path)
    #         }
    #     }
        
    #     try:
    #         print(f"Starting evaluation for {model_name}")
            
    #         # Stage 1: Load Model
    #         if progress_callback:
    #             progress_callback(self.update_progress(1))
            
    #         model, error = self.load_model(model_file_path)
    #         if error:
    #             results['error'] = error
    #             print(f"Model loading failed: {error}")
    #             return results
            
    #         results['files_processed'] += 1
            
    #         # Stage 2: Load Test Data
    #         if progress_callback:
    #             progress_callback(self.update_progress(2))
            
    #         X_test, y_test, error = self.load_test_data(test_file_path, steps_path)
    #         print(f'X_test:{X_test.head()}')
    #         print('---------------')
    #         print(f'y_test:{y_test.head()}')
    #         if error:
    #             results['error'] = error
    #             print(f"Test data loading failed: {error}")
    #             return results
            
    #         results['files_processed'] += 1
            
    #         # Stage 3: Make Predictions
    #         if progress_callback:
    #             progress_callback(self.update_progress(3))
            
    #         try:
    #             print("Making predictions...")
    #             y_pred = model.predict(X_test)
    #             print(f'y_pred: {y_pred[:5]}... (showing first 5 predictions)')
    #             print(f"Predictions completed: {len(y_pred)} predictions made")
    #         except Exception as e:
    #             error_msg = f"Prediction failed: {str(e)}"
    #             results['error'] = error_msg
    #             print(error_msg)
    #             return results
            
    #         # Stage 4: Calculate Metrics
    #         if progress_callback:
    #             progress_callback(self.update_progress(4))
            
    #         problem_type = get_problem_type_from_model(model)
    #         results['problem_type'] = problem_type
    #         print(f"Problem type from model: {problem_type}")
            
    #         if problem_type == 'classification':
    #             metrics, error = calculate_detailed_metrics(y_test, y_pred, problem_type, model, X_test)
    #             if error:
    #                 results['error'] = error
    #                 print(f"Classification metrics calculation failed: {error}")
    #                 return results
    #             correct_predictions = sum(1 for actual, pred in zip(y_test, y_pred) if actual == pred)
    #             results['accuracy_count'] = correct_predictions
    #             results['success_rate'] = (correct_predictions / len(y_test)) * 100
    #             print(f"Classification results: {correct_predictions}/{len(y_test)} correct ({results['success_rate']:.2f}%)")
    #         else:  # regression
    #             metrics, error = calculate_detailed_metrics(y_test, y_pred, problem_type, model, X_test)
    #             if error:
    #                 results['error'] = error
    #                 print(f"Regression metrics calculation failed: {error}")
    #                 return results
    #             results['mean_error'] = float(np.mean([abs(float(actual) - float(pred)) 
    #                                                  for actual, pred in zip(y_test, y_pred)]))
    #             print(f"Regression results: Mean error = {results['mean_error']:.4f}")
            
    #         results['metrics'] = metrics
            
    #         # Create detailed predictions
    #         predictions_data = self.create_predictions_data(y_test, y_pred, problem_type)
    #         results['predictions'] = predictions_data
    #         results['total_tests'] = len(predictions_data)
            
    #         # Stage 5: Finalize
    #         if progress_callback:
    #             progress_callback(self.update_progress(5))
            
    #         print(f"Evaluation completed successfully for {model_name}")
    #         return results
            
    #     except Exception as e:
    #         error_msg = f"Evaluation failed: {str(e)}"
    #         results['error'] = error_msg
    #         print(f"{error_msg}\n{traceback.format_exc()}")
    #         return results
    def run_evaluation(self, model_name, model_file_path, test_file_path, steps_path=None, progress_callback=None):
        """Run complete ML model evaluation with comprehensive error handling"""
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'files_processed': 0,
            'file_info': {
                'model_file': os.path.basename(model_file_path),
                'test_file': os.path.basename(test_file_path)
            }
        }
        
        try:
            print(f"Starting evaluation for {model_name}")
            
            # Stage 1: Load Model
            if progress_callback:
                progress_callback(self.update_progress(1))
            
            model, error = self.load_model(model_file_path)
            if error:
                results['error'] = error
                print(f"Model loading failed: {error}")
                return results
            
            results['files_processed'] += 1
            
            # Stage 2: Load Test Data
            if progress_callback:
                progress_callback(self.update_progress(2))
            
            X_test, y_test, error = self.load_test_data(test_file_path, steps_path)
            print(f'X_test:{X_test.head()}')
            print('---------------')
            print(f'y_test:{y_test.head()}')
            if error:
                results['error'] = error
                print(f"Test data loading failed: {error}")
                return results
            
            results['files_processed'] += 1
            
            # Stage 3: Make Predictions
            if progress_callback:
                progress_callback(self.update_progress(3))
            
            try:
                print("Making predictions...")
                y_pred = model.predict(X_test)
                print(f'y_pred: {y_pred[:5]}... (showing first 5 predictions)')
                print(f"Predictions completed: {len(y_pred)} predictions made")
            except Exception as e:
                error_msg = f"Prediction failed: {str(e)}"
                results['error'] = error_msg
                print(error_msg)
                return results
            
            # Stage 4: Calculate Metrics
            if progress_callback:
                progress_callback(self.update_progress(4))
            
            problem_type = get_problem_type_from_model(model)
            results['problem_type'] = problem_type
            print(f"Problem type from model: {problem_type}")
            
            if problem_type == 'classification':
                metrics, error = calculate_detailed_metrics(y_test, y_pred, problem_type, model, X_test)
                if error:
                    results['error'] = error
                    print(f"Classification metrics calculation failed: {error}")
                    return results
                correct_predictions = sum(1 for actual, pred in zip(y_test, y_pred) if actual == pred)
                results['accuracy_count'] = correct_predictions
                results['success_rate'] = (correct_predictions / len(y_test)) * 100
                print(f"Classification results: {correct_predictions}/{len(y_test)} correct ({results['success_rate']:.2f}%)")
            else:  # regression
                y_test = pd.to_numeric(y_test, errors='coerce')
                y_pred = pd.to_numeric(y_pred, errors='coerce')
                print("After numeric conversion, y_test head:", y_test.head())
                print("After numeric conversion, y_pred head:", y_pred[:5])    
                metrics, error = calculate_detailed_metrics(y_test, y_pred, problem_type, model, X_test)
                if error:
                    results['error'] = error
                    print(f"Regression metrics calculation failed: {error}")
                    return results
                results['mean_error'] = float(np.mean([abs(float(actual) - float(pred)) 
                                                    for actual, pred in zip(y_test, y_pred)]))
                print(f"Regression results: Mean error = {results['mean_error']:.4f}")
            
            results['metrics'] = metrics
            
            # Create detailed predictions with original test data
            predictions_data = self.create_predictions_data(y_test, y_pred, problem_type, X_test)
            results['predictions'] = predictions_data
            results['total_tests'] = len(predictions_data)
            
            # Stage 5: Finalize
            if progress_callback:
                progress_callback(self.update_progress(5))
            
            print(f"Evaluation completed successfully for {model_name}")
            return results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            results['error'] = error_msg
            print(f"{error_msg}\n{traceback.format_exc()}")
            return results
    

def run_custom_ml_evaluation_task(model_name, model_file_path, test_file_path, steps_path=None):
    """
    Run custom ML evaluation task with improved error handling and progress tracking
    """
    print(f"Starting evaluation task for {model_name}")
    
    def progress_callback(stage_info):
        """Progress callback with error handling"""
        try:
            custom_evaluation_progress[model_name] = stage_info
            processing_status[f"{model_name}_ml_custom"] = "processing"
            print(f"Progress update for {model_name}: Stage {stage_info.get('stage', 'unknown')} - {stage_info.get('message', 'No message')}")
        except Exception as e:
            print(f"Error in progress callback: {str(e)}")
    
    try:
        # Initialize evaluator
        evaluator = CustomMLEvaluator()
        print(f"CustomMLEvaluator initialized for {model_name}")
        
        # Run evaluation with progress callback
        results = evaluator.run_evaluation(
            model_name, 
            model_file_path, 
            test_file_path, 
            steps_path, 
            progress_callback
        )
        
        # Store results
        custom_evaluation_results[f"{model_name}_ml"] = results
        
        # Save CSV and Excel files immediately after evaluation
        output_dir = os.path.join("uploads", model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(output_dir, f"{model_name}_test_results.csv")
        if results.get("predictions"):
            pd.DataFrame(results["predictions"]).to_csv(csv_path, index=False)

        # Save Excel
        excel_path = os.path.join(output_dir, f"{model_name}_custom_ml_report.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if "predictions" in results:
                df_predictions = pd.DataFrame(results["predictions"])
                df_predictions.to_excel(writer, sheet_name="Test_Results", index=False)
            metrics = results.get("metrics", {})
            summary_data = {
                "Metric": ["Model Name", "Evaluation Date", "Problem Type", "Total Tests"] + list(metrics.keys()),
                "Value": [
                    results.get("model_name", model_name),
                    results.get("timestamp", "N/A"),
                    results.get("problem_type", "Unknown"),
                    results.get("total_tests", 0)
                ] + list(metrics.values())
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Update status based on results
        if results.get("error"):
            processing_status[f"{model_name}_ml_custom"] = "error"
            print(f"Evaluation failed for {model_name}: {results['error']}")
        else:
            processing_status[f"{model_name}_ml_custom"] = "complete"
            print(f"Evaluation completed successfully for {model_name}")
            print(f"Results summary: {results.get('problem_type', 'Unknown')} problem with {results.get('total_tests', 0)} test cases")
        
        return results
        
    except Exception as e:
        error_msg = f"Evaluation task failed for {model_name}: {str(e)}"
        print(error_msg)
        
        # Store error in results
        error_results = {
            'error': error_msg,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'files_processed': 0
        }
        
        custom_evaluation_results[f"{model_name}_ml"] = error_results
        processing_status[f"{model_name}_ml_custom"] = "error"
        
        return error_results

def get_problem_type_from_model(model):
    """Determine problem type from model class."""
    from sklearn.base import is_classifier, is_regressor
    if is_classifier(model):
        return 'classification'
    elif is_regressor(model):
        return 'regression'
    else:
        return 'unknown'
    
def get_custom_ml_status(model_name):
    """Get custom ML evaluation status with improved error handling"""
    try:
        status = processing_status.get(f"{model_name}_ml_custom", "not_started")
        results = custom_evaluation_results.get(f"{model_name}_ml", {})
        progress = custom_evaluation_progress.get(model_name, {})
        
        # Add additional debug info
        status_data = {
            "status": status,
            "results": results,
            "progress": progress,
            "timestamp": datetime.now().isoformat(),
            "debug_info": {
                "has_results": bool(results),
                "has_progress": bool(progress),
                "results_keys": list(results.keys()) if results else [],
                "processing_keys": list(processing_status.keys())
            }
        }
        
        return status_data
        
    except Exception as e:
        print(f"Error getting status for {model_name}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def clear_custom_ml_results(model_name):
    custom_evaluation_results.pop(f"{model_name}_ml", None)
    processing_status.pop(f"{model_name}_ml_custom", None)
    return True


def export_custom_ml_excel(model_name):
    results = get_custom_ml_status(model_name)
    if not results or results.get('error'):
        return None, "No evaluation results found."
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Predictions sheet
        if 'predictions' in results:
            df_predictions = pd.DataFrame(results['predictions'])
            df_predictions.to_excel(writer, sheet_name='Test_Results', index=False)
        # Summary sheet
        metrics = results.get('metrics', {})
        summary_data = {
            'Metric': ['Model Name', 'Evaluation Date', 'Problem Type', 'Total Tests'] + list(metrics.keys()),
            'Value': [
                results.get('model_name', model_name),
                results.get('timestamp', 'N/A'),
                results.get('problem_type', 'Unknown'),
                results.get('total_tests', 0)
            ] + list(metrics.values())
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    output.seek(0)
    return output, None

def export_custom_ml_csv(model_name):
    results = get_custom_ml_status(model_name)
    if not results or not results.get('predictions'):
        return None, "No test results available."
    from io import BytesIO
    output = BytesIO()
    df = pd.DataFrame(results['predictions'])
    df.to_csv(output, index=False)
    output.seek(0)
    return output, None

def calculate_detailed_metrics(y_true, y_pred, problem_type, model=None, X_test=None):
    """Calculate comprehensive evaluation metrics (ported from evaluate_ml_supervised_mlflow.py)."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix, mean_absolute_percentage_error
    )
    from scipy import stats

    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics = {}

    # Convert to numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove NaNs for regression
    if problem_type == 'regression':
        y_true = pd.to_numeric(y_true, errors='coerce')
        y_pred = pd.to_numeric(y_pred, errors='coerce')
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    # Add this check:
    if len(y_true) == 0 or len(y_pred) == 0:
        return None, "No valid numeric values found for regression metrics"

    if problem_type == 'regression':
        metrics['mae'] = convert_to_serializable(mean_absolute_error(y_true, y_pred))
        metrics['mse'] = convert_to_serializable(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = convert_to_serializable(np.sqrt(metrics['mse']))
        metrics['r2'] = convert_to_serializable(r2_score(y_true, y_pred))
        try:
            metrics['mape'] = convert_to_serializable(mean_absolute_percentage_error(y_true, y_pred))
        except:
            metrics['mape'] = None
        metrics['max_error'] = convert_to_serializable(np.max(np.abs(y_true - y_pred)))
        metrics['mean_error'] = convert_to_serializable(np.mean(y_true - y_pred))
        metrics['std_error'] = convert_to_serializable(np.std(y_true - y_pred))
        # Statistical tests
        residuals = y_true - y_pred
        try:
            _, p_value = stats.normaltest(residuals)
            metrics['residuals_normality_p'] = convert_to_serializable(p_value)
        except Exception:
            metrics['residuals_normality_p'] = None
        try:
            metrics['explained_variance'] = convert_to_serializable(1 - (np.var(residuals) / np.var(y_true)))
        except Exception:
            metrics['explained_variance'] = None
    else:  # classification
        metrics['accuracy'] = convert_to_serializable(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = convert_to_serializable(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['precision_weighted'] = convert_to_serializable(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_macro'] = convert_to_serializable(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_weighted'] = convert_to_serializable(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_macro'] = convert_to_serializable(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_weighted'] = convert_to_serializable(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        # Per-class metrics
        unique_classes = np.unique(y_true)
        if len(unique_classes) <= 10:
            class_report = classification_report(y_true, y_pred, output_dict=True)
            for class_name in [str(c) for c in unique_classes]:
                if class_name in class_report:
                    metrics[f'precision_class_{class_name}'] = convert_to_serializable(class_report[class_name]['precision'])
                    metrics[f'recall_class_{class_name}'] = convert_to_serializable(class_report[class_name]['recall'])
                    metrics[f'f1_class_{class_name}'] = convert_to_serializable(class_report[class_name]['f1-score'])
        # Confusion matrix stats
        cm = confusion_matrix(y_true, y_pred)
        if len(unique_classes) == 2 and cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = convert_to_serializable(tn)
            metrics['false_positives'] = convert_to_serializable(fp)
            metrics['false_negatives'] = convert_to_serializable(fn)
            metrics['true_positives'] = convert_to_serializable(tp)
            metrics['specificity'] = convert_to_serializable(tn / (tn + fp) if (tn + fp) > 0 else 0)
            metrics['sensitivity'] = convert_to_serializable(tp / (tp + fn) if (tp + fn) > 0 else 0)
    
    return metrics, None

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj