import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .create_plots_ml_supervised import (
    generate_model_summary_plots, create_regression_plots, create_classification_plots)
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn') 
sns.set_palette("husl")

# Global variables for progress tracking
ml_evaluation_progress = {}
ml_evaluation_results = {}

def update_progress(model_name, current_task, progress_percent):
    """Update progress for a specific model evaluation."""
    ml_evaluation_progress[model_name] = {
        "current_task": current_task,
        "progress_percent": progress_percent,
        "timestamp": datetime.now().isoformat()
    }

def get_ml_progress(model_name):
    """Get current progress for a model evaluation."""
    return ml_evaluation_progress.get(model_name, {
        "current_task": "Not started",
        "progress_percent": 0,
        "timestamp": datetime.now().isoformat()
    })

def clear_ml_progress(model_name):
    """Clear progress for a specific model."""
    if model_name in ml_evaluation_progress:
        del ml_evaluation_progress[model_name]
    if model_name in ml_evaluation_results:
        del ml_evaluation_results[model_name]

def load_model(model_path):
    """Load pickle model."""
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
def detect_problem_type(model):
    model_type = type(model).__name__.lower()
    # Check regression keywords first!
    regression_keywords = ['regression', 'regressor', 'linear', 'ridge', 'lasso', 'elastic']
    classification_keywords = ['classifier', 'logistic', 'svm', 'randomforest', 'decisiontree', 
                               'gradient', 'xgb', 'lgb', 'naive', 'knn', 'ada', 'extra', 'voting']
    for keyword in regression_keywords:
        if keyword in model_type:
            print(f"Detected regression model: {model_type}")
            return 'regression'
    for keyword in classification_keywords:
        if keyword in model_type:
            print(f"Detected classification model: {model_type}")
            return 'classification'
    print('--------------------------------------------')
    # Default to regression if uncertain
    return 'regression'

# def detect_problem_type(model):
#     """Detect if it's classification or regression based on model type."""
#     model_type = type(model).__name__.lower()
    
#     # Common classification models
#     classification_keywords = ['classifier', 'logistic', 'svm', 'randomforest', 'decisiontree', 
#                              'gradient', 'xgb', 'lgb', 'naive', 'knn', 'ada', 'extra', 'voting']
    
#     # Common regression models  
#     regression_keywords = ['regression', 'regressor', 'linear', 'ridge', 'lasso', 'elastic']
    
#     for keyword in classification_keywords:
#         if keyword in model_type:
#             print(f"Detected classification model: {model_type}")
#             return 'classification'
    
#     for keyword in regression_keywords:
#         if keyword in model_type:
#             print(f"Detected regression model: {model_type}")
#             return 'regression'
#     print('--------------------------------------------')
#     # Default to regression if uncertain
#     return 'regression'

def round_if_needed(preds, model_name):
    """Round predictions if needed based on model name."""
    if any(keyword in model_name.lower() for keyword in ['rating', 'ordinal', 'score', 'rank']):
        return np.clip(np.round(preds).astype(int), 0, 10)
    return preds

def get_model_info(model):
    """Extract comprehensive model information."""
    info = {
        'model_type': type(model).__name__,
        'model_params': getattr(model, 'get_params', lambda: {})(),
        'feature_count': None,
        'has_feature_importance': hasattr(model, 'feature_importances_'),
        'has_predict_proba': hasattr(model, 'predict_proba'),
        'has_coefficients': hasattr(model, 'coef_'),
        'has_intercept': hasattr(model, 'intercept_'),
        'training_score': None
    }
    
    # Try to get feature count from model
    if hasattr(model, 'n_features_in_'):
        info['feature_count'] = model.n_features_in_
    elif hasattr(model, 'coef_'):
        if hasattr(model.coef_, 'shape'):
            info['feature_count'] = model.coef_.shape[-1] if model.coef_.ndim > 1 else len(model.coef_)
    
    # Get training score if available
    if hasattr(model, 'score_'):
        info['training_score'] = model.score_
    print(f"Model Info: {info}")
    return info

def cleanup_evaluation_resources(model_name):
    """Clean up all resources after evaluation."""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        
        # Clear any cached data
        import gc
        gc.collect()
        
        print(f"Cleaned up resources for {model_name}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


def calculate_detailed_metrics(y_true, y_pred, problem_type, model=None, X_test=None):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    if problem_type == 'regression':
        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional regression metrics
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            metrics['mape'] = None
        
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['mean_error'] = np.mean(y_true - y_pred)
        metrics['std_error'] = np.std(y_true - y_pred)
        
        # Statistical tests
        residuals = y_true - y_pred
        _, p_value = stats.normaltest(residuals)
        metrics['residuals_normality_p'] = p_value
        
        # Explained variance
        metrics['explained_variance'] = 1 - (np.var(residuals) / np.var(y_true))
        
    else:  # classification
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        unique_classes = np.unique(y_true)
        if len(unique_classes) <= 10:  # Only for reasonable number of classes
            class_report = classification_report(y_true, y_pred, output_dict=True)
            for class_name in [str(c) for c in unique_classes]:
                if class_name in class_report:
                    metrics[f'precision_class_{class_name}'] = class_report[class_name]['precision']
                    metrics[f'recall_class_{class_name}'] = class_report[class_name]['recall']
                    metrics[f'f1_class_{class_name}'] = class_report[class_name]['f1-score']
        
        # ROC AUC for binary classification
        if len(unique_classes) == 2 and model is not None and hasattr(model, 'predict_proba') and X_test is not None:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = None
        
        # Confusion matrix stats
        cm = confusion_matrix(y_true, y_pred)
        if len(unique_classes) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

def run_ml_evaluation(model_name, model_path, dataset_path):
    """Run comprehensive ML model evaluation using MLflow."""
    try:
        update_progress(model_name, "Initializing MLflow...", 5)
        
        # Set up MLflow
        mlflow.set_experiment(f"ML_Evaluation_{model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            update_progress(model_name, "Loading model...", 15)
            
            # Load model directly from the provided path
            model = load_model(model_path)
            if model is None:
                raise Exception("Failed to load model")
            
            update_progress(model_name, "Loading test data...", 25)
            
            # Load and validate test data
            if not os.path.exists(dataset_path):
                raise Exception(f"Test dataset not found: {dataset_path}")
            
            df = pd.read_csv(dataset_path)
            print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
            if 'target' not in df.columns:
                # If no 'target' column, assume last column is target
                target_col = df.columns[-1]
                df = df.rename(columns={target_col: 'target'})
                print(f"No 'target' column found, using '{target_col}' as target")
            
            X_test = df.drop(columns=['target'])
            y_test = df['target'].values
            
            update_progress(model_name, "Analyzing model...", 35)
            # Get model information
            model_info = get_model_info(model)
            problem_type = detect_problem_type(model)
            
            # Log model info to MLflow
            mlflow.log_param("model_type", model_info['model_type'])
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("feature_count", model_info['feature_count'])
            mlflow.log_param("has_feature_importance", model_info['has_feature_importance'])
            mlflow.log_param("has_predict_proba", model_info['has_predict_proba'])
            mlflow.log_param("has_coefficients", model_info['has_coefficients'])
            
            # Log model parameters
            for param_name, param_value in model_info['model_params'].items():
                try:
                    print(f"Logging parameter: {param_name} = {param_value}")
                    if isinstance(param_value, (int, float, str, bool)):
                        mlflow.log_param(f"model_{param_name}", param_value)
                except:
                    pass  # Skip parameters that can't be logged
            
            update_progress(model_name, "Making predictions...", 45)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Round predictions if needed (for rating/ordinal models)
            y_pred = round_if_needed(y_pred, model_name)
            
            update_progress(model_name, "Calculating metrics...", 55)
            
            # Calculate comprehensive metrics
            metrics = calculate_detailed_metrics(y_test, y_pred, problem_type, model, X_test)
            print(f"Calculated metrics: {metrics}")
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    try:
                        mlflow.log_metric(metric_name, metric_value)
                    except (TypeError, ValueError):
                        pass  # Skip non-numeric metrics
            
            update_progress(model_name, "Generating visualizations...", 70)
            
            # Create plots directories
            plots_dir = f"static/plots/{model_name}"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate model summary plots
            summary_plots = generate_model_summary_plots(model_name, model, model_info)
            
            # Generate evaluation plots based on problem type
            if problem_type == 'regression':
                eval_plots = create_regression_plots(model_name, y_test, y_pred, plots_dir)
            else:
                eval_plots = create_classification_plots(model_name, model, X_test, y_test, y_pred, plots_dir)
            
            plt.close('all')
            # Combine all plots
            all_plots = {**summary_plots, **eval_plots}
            
            update_progress(model_name, "Logging artifacts...", 85)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log plots as artifacts
            for plot_name, plot_path in all_plots.items():
                # Don't modify the path if it's already correct
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, "plots")
                else:
                    # Try with static prefix
                    actual_path = f"static/{plot_path}" if not plot_path.startswith("static/") else plot_path
                    if os.path.exists(actual_path):
                        mlflow.log_artifact(actual_path, "plots")
            
            # Save predictions and actual values
            results_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'residuals': y_test - y_pred if problem_type == 'regression' else None
            })
            print(f"Saving predictions to {plots_dir}/predictions.csv")
            print(f'results_df: {results_df.head()}')
            results_path = f"{plots_dir}/predictions.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path, "data")
            
            update_progress(model_name, "Finalizing results...", 95)
            
            # Prepare final results
            results = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'problem_type': problem_type,
                'model_info': model_info,
                'metrics': metrics,
                'plots': all_plots,
                'mlflow_run_id': mlflow.active_run().info.run_id,
                'model_path': model_path,
                'dataset_path': dataset_path,
                'num_test_samples': len(y_test),
                'num_features': X_test.shape[1]
            }
            
            # Store results
            ml_evaluation_results[model_name] = results
            
            update_progress(model_name, "Evaluation completed!", 100)
            
            return results
            
    except Exception as e:
        error_msg = str(e)
        update_progress(model_name, f"Error: {error_msg}", 0)
        
        # Store error results with proper structure
        error_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'model_info': {  # Provide default model_info structure
                'model_type': 'Unknown',
                'has_feature_importance': False,
                'has_predict_proba': False,
                'has_coefficients': False,
                'feature_count': None,
                'model_params': {}
            },
            'problem_type': 'unknown',
            'metrics': {},
            'plots': {}
        }
        ml_evaluation_results[model_name] = error_results
        raise e
    finally:
        # Always clean up resources
        cleanup_evaluation_resources(model_name)

def get_ml_results(model_name):
    """Get evaluation results for a specific model."""
    return ml_evaluation_results.get(model_name, {})

def list_available_results():
    """List all available evaluation results."""
    return list(ml_evaluation_results.keys())

def cleanup_ml_resources(model_name=None):
    """Clean up resources for a specific model or all models."""
    if model_name:
        # Clean up specific model resources
        plots_dir = f"static/plots/{model_name}"
        if os.path.exists(plots_dir):
            import shutil
            shutil.rmtree(plots_dir)
        clear_ml_progress(model_name)
    else:
        # Clean up all resources
        if os.path.exists("static/plots"):
            import shutil
            shutil.rmtree("static/plots")
        ml_evaluation_progress.clear()
        ml_evaluation_results.clear()


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


def export_results_to_json(model_name, output_path=None):
    """Export evaluation results to JSON file with better error handling."""
    if model_name not in ml_evaluation_results:
        raise ValueError(f"No results found for model: {model_name}")
    
    try:
        results = ml_evaluation_results[model_name].copy()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results = convert_numpy_types(results)
        
        # Always use a predictable filename for download
        if output_path is None:
            output_dir = f"results/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"ml_evaluation_{model_name}.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return output_path
        
    except Exception as e:
        print(f"Error exporting results to JSON: {e}")
        raise e

def compare_models(model_names, metric='accuracy'):
    """Compare multiple models based on a specific metric."""
    comparison_data = []
    
    for model_name in model_names:
        if model_name in ml_evaluation_results:
            results = ml_evaluation_results[model_name]
            if 'metrics' in results and metric in results['metrics']:
                comparison_data.append({
                    'model_name': model_name,
                    'metric_value': results['metrics'][metric],
                    'problem_type': results.get('problem_type', 'unknown'),
                    'timestamp': results.get('timestamp', '')
                })
    
    # Sort by metric value (descending for most metrics)
    comparison_data.sort(key=lambda x: x['metric_value'], reverse=True)
    
    return comparison_data

# # Main execution function for command line usage
# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) != 4:
#         print("Usage: python ml_evaluation.py <model_name> <model_path> <dataset_path>")
#         sys.exit(1)
    
#     model_name = sys.argv[1]
#     model_path = sys.argv[2]
#     dataset_path = sys.argv[3]
    
#     try:
#         print(f"Starting evaluation for model: {model_name}")
#         results = run_ml_evaluation(model_name, model_path, dataset_path)
        
#         print(f"\nEvaluation completed successfully!")
#         print(f"Problem type: {results['problem_type']}")
#         print(f"Model type: {results['model_info']['model_type']}")
#         print(f"Number of features: {results['num_features']}")
#         print(f"Test samples: {results['num_test_samples']}")
        
#         if 'metrics' in results:
#             print(f"\nKey Metrics:")
#             for metric, value in list(results['metrics'].items())[:5]:  # Show first 5 metrics
#                 if value is not None:
#                     print(f"  {metric}: {value:.4f}")
        
#         print(f"\nMLflow Run ID: {results['mlflow_run_id']}")
#         print(f"Plots generated: {len(results['plots'])}")
        
#         # Export results to JSON
#         json_path = export_results_to_json(model_name)
#         print(f"Results exported to: {json_path}")
        
#     except Exception as e:
#         print(f"Evaluation failed: {str(e)}")
#         sys.exit(1)

def run_ml_evaluation_wrapper(model_name, model_file_path, test_csv_path, benchmark):
    """Wrapper function to run ML evaluation in background thread."""
    try:
        # Ensure matplotlib uses non-GUI backend in thread
        import matplotlib
        matplotlib.use('Agg')
        
        print(f"Starting ML evaluation for {model_name}")
        print(f"Model file: {model_file_path}")
        print(f"Test data: {test_csv_path}")
        
        # Run the evaluation
        results = run_ml_evaluation(model_name, model_file_path, test_csv_path)
        
        print(f"ML evaluation completed for {model_name}")
        
        # Clean up matplotlib resources
        import matplotlib.pyplot as plt
        plt.close('all')
        
    except Exception as e:
        print(f"Error in ML evaluation thread: {e}")
        update_progress(model_name, f"Error: {str(e)}", 0)
        # Clean up on error
        import matplotlib.pyplot as plt
        plt.close('all')
    

def extract_test_csv_if_needed(dataset_path):
    """Extract test.csv from dataset.zip if it doesn't exist."""
    test_csv_path = os.path.join(dataset_path, 'test.csv')
    dataset_zip_path = os.path.join(dataset_path, 'dataset.zip')
    
    # If test.csv already exists, return its path
    if os.path.exists(test_csv_path):
        return test_csv_path
    
    # If dataset.zip exists, try to extract test.csv
    if os.path.exists(dataset_zip_path):
        try:
            import zipfile
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                # Look for test.csv in the zip
                if 'test.csv' in zip_ref.namelist():
                    zip_ref.extract('test.csv', dataset_path)
                    return test_csv_path
                else:
                    # Look for any CSV file that might be the test data
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if csv_files:
                        # Extract the first CSV file and rename it to test.csv
                        zip_ref.extract(csv_files[0], dataset_path)
                        extracted_path = os.path.join(dataset_path, csv_files[0])
                        os.rename(extracted_path, test_csv_path)
                        return test_csv_path
        except Exception as e:
            print(f"Error extracting from dataset.zip: {e}")
    
    return None

def generate_ml_report(results):
    """Generate comprehensive ML evaluation report."""
    report = {
        'model_name': results.get('model_name'),
        'evaluation_timestamp': results.get('timestamp'),
        'problem_type': results.get('problem_type'),
        'dataset_info': results.get('dataset_info'),
        'performance_metrics': results.get('metrics'),
        'mlflow_run_id': results.get('mlflow_run_id'),
        'summary': {
            'evaluation_completed': True,
            'total_samples': results.get('dataset_info', {}).get('n_samples', 0),
            'total_features': results.get('dataset_info', {}).get('n_features', 0)
        }
    }
    
    # Add key performance indicators
    metrics = results.get('metrics', {})
    if results.get('problem_type') == 'classification':
        report['summary']['key_metrics'] = {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0)
        }
        if 'roc_auc' in metrics:
            report['summary']['key_metrics']['roc_auc'] = metrics['roc_auc']
    else:
        report['summary']['key_metrics'] = {
            'r2_score': metrics.get('r2_score', 0),
            'rmse': metrics.get('rmse', 0),
            'mae': metrics.get('mae', 0),
            'mape': metrics.get('mean_absolute_percentage_error', 0)
        }
    
    # Add cross-validation summary
    if 'cv_scores' in metrics and metrics['cv_scores']:
        report['cross_validation'] = {
            'mean_score': metrics['cv_scores']['mean'],
            'std_score': metrics['cv_scores']['std'],
            'individual_scores': metrics['cv_scores']['scores']
        }
    
    return json.dumps(report, indent=2, default=str)