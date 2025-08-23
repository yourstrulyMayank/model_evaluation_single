import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

import contextlib
import threading

plot_lock = threading.Lock()

@contextlib.contextmanager
def thread_safe_plotting():
    """Context manager for thread-safe matplotlib operations."""
    with plot_lock:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            yield plt
        finally:
            plt.close('all')

def create_regression_plots(model_name, y_true, y_pred, plots_dir):
    """Create comprehensive regression evaluation plots."""
    plots = {}
    with thread_safe_plotting() as plt:
    
        # 1. Actual vs Predicted Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, c='blue', s=50)
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Trend line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), "g--", alpha=0.8, label=f'Trend Line (slope={z[0]:.3f})')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'Actual vs Predicted Values - {model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        actual_pred_path = f"{plots_dir}/actual_vs_predicted.png"
        plt.savefig(actual_pred_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['actual_vs_predicted'] = f"plots/{model_name}/actual_vs_predicted.png"
        
        # 2. Residuals Plot
        residuals = y_true - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, c='red', s=50)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residuals Distribution
        ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(residuals.mean(), color='red', linestyle='--', 
                    label=f'Mean: {residuals.mean():.4f}')
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        residuals_path = f"{plots_dir}/residuals_analysis.png"
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['residuals_analysis'] = f"plots/{model_name}/residuals_analysis.png"
        
        # 3. Error Distribution Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = np.abs(residuals)
        
        ax.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(errors.mean(), color='red', linestyle='--', 
                label=f'Mean Absolute Error: {errors.mean():.4f}')
        ax.axvline(np.median(errors), color='green', linestyle='--', 
                label=f'Median Absolute Error: {np.median(errors):.4f}')
        
        ax.set_xlabel('Absolute Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Error Distribution - {model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        error_dist_path = f"{plots_dir}/error_distribution.png"
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['error_distribution'] = f"plots/{model_name}/error_distribution.png"
    
    return plots

def create_classification_plots(model_name, model, X_test, y_true, y_pred, plots_dir):
    """Create comprehensive classification evaluation plots."""
    plots = {}
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    cm_path = f"{plots_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots['confusion_matrix'] = f"plots/{model_name}/confusion_matrix.png"
    
    # 2. Classification Report Heatmap
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='RdYlBu_r', 
                square=True, linewidths=0.5, fmt='.3f')
    ax.set_title(f'Classification Report - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    report_path = f"{plots_dir}/classification_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots['classification_report'] = f"plots/{model_name}/classification_report.png"
    
    # 3. ROC Curve (for binary classification)
    unique_classes = len(np.unique(y_true))
    if unique_classes == 2 and hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC Curve (AUC = {auc_score:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            roc_path = f"{plots_dir}/roc_curve.png"
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['roc_curve'] = f"plots/{model_name}/roc_curve.png"
            
            # 4. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(recall, precision, color='darkorange', lw=2, 
                   label=f'PR Curve')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pr_path = f"{plots_dir}/precision_recall_curve.png"
            plt.savefig(pr_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['precision_recall_curve'] = f"plots/{model_name}/precision_recall_curve.png"
            
        except Exception as e:
            print(f"Could not generate ROC/PR curves: {e}")
    
    # 4. Class Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True class distribution
    true_counts = pd.Series(y_true).value_counts().sort_index()
    ax1.bar(true_counts.index, true_counts.values, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Predicted class distribution
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    ax2.bar(pred_counts.index, pred_counts.values, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = f"{plots_dir}/class_distribution.png"
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots['class_distribution'] = f"plots/{model_name}/class_distribution.png"
    
    return plots

def generate_model_summary_plots(model_name, model, model_info):
    """Generate enhanced model summary visualizations."""
    plots = {}
    
    try:
        plots_dir = f"static/plots/{model_name}"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Enhanced Model Parameters Visualization
        if model_info['model_params']:
            fig, ax = plt.subplots(figsize=(12, 8))
            params = model_info['model_params']
            
            # Filter and categorize parameters
            numeric_params = {k: v for k, v in params.items() 
                            if isinstance(v, (int, float)) and not isinstance(v, bool)}
            boolean_params = {k: v for k, v in params.items() if isinstance(v, bool)}
            string_params = {k: v for k, v in params.items() if isinstance(v, str)}
            
            if numeric_params:
                param_names = list(numeric_params.keys())
                param_values = list(numeric_params.values())
                
                # Create color map based on parameter values
                colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
                bars = ax.bar(param_names, param_values, color=colors, alpha=0.8)
                
                ax.set_title(f'Model Parameters - {model_name}', fontsize=16, fontweight='bold')
                ax.set_ylabel('Parameter Values', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, param_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}' if isinstance(value, float) else str(value),
                           ha='center', va='bottom', fontsize=10)
                
                # Add parameter summary text
                param_text = f"Total Parameters: {len(params)}\n"
                param_text += f"Numeric: {len(numeric_params)}\n"
                param_text += f"Boolean: {len(boolean_params)}\n"
                param_text += f"String: {len(string_params)}"
                
                ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                       fontsize=10, verticalalignment='top')
                
                plt.tight_layout()
                param_path = f"{plots_dir}/model_parameters.png"
                plt.savefig(param_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots['model_parameters'] = f"plots/{model_name}/model_parameters.png"
        
        # 2. Enhanced Feature Importance
        if model_info['has_feature_importance']:
            try:
                importances = model.feature_importances_
                if len(importances) <= 50:  # Reasonable number of features
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Top features bar plot
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
                    indices = np.argsort(importances)[::-1][:min(15, len(importances))]
                    
                    bars = ax1.bar(range(len(indices)), importances[indices], 
                                  color='lightcoral', alpha=0.8)
                    ax1.set_title(f'Top {len(indices)} Feature Importances', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Importance Score', fontsize=12)
                    ax1.set_xticks(range(len(indices)))
                    ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                    
                    for bar, idx in zip(bars, indices):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                               f'{importances[idx]:.4f}', ha='center', va='bottom', fontsize=9)
                    
                    # Cumulative importance
                    sorted_importances = np.sort(importances)[::-1]
                    cumulative_importances = np.cumsum(sorted_importances)
                    
                    ax2.plot(range(1, len(cumulative_importances) + 1), cumulative_importances, 
                            'b-', linewidth=2, marker='o', markersize=4)
                    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, 
                               label='95% Threshold')
                    ax2.set_xlabel('Number of Features', fontsize=12)
                    ax2.set_ylabel('Cumulative Importance', fontsize=12)
                    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    feat_path = f"{plots_dir}/feature_importance.png"
                    plt.savefig(feat_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots['feature_importance'] = f"plots/{model_name}/feature_importance.png"
            except Exception as e:
                print(f"Could not generate feature importance plot: {e}")
        
        # 3. Enhanced Coefficients Plot (for linear models)
        if model_info['has_coefficients']:
            try:
                coef = model.coef_
                if hasattr(coef, 'flatten'):
                    coef = coef.flatten()
                
                if len(coef) <= 50:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Coefficients bar plot
                    feature_names = [f'Feature_{i}' for i in range(len(coef))]
                    colors = ['red' if c < 0 else 'blue' for c in coef]
                    
                    bars = ax1.bar(feature_names, coef, color=colors, alpha=0.7)
                    ax1.set_title(f'Model Coefficients - {model_name}', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Coefficient Value', fontsize=12)
                    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                    
                    for bar, value in zip(bars, coef):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., 
                               height + (0.01 * max(abs(coef)) if height >= 0 else -0.01 * max(abs(coef))),
                               f'{value:.3f}', ha='center', 
                               va='bottom' if height >= 0 else 'top', fontsize=9)
                    
                    # Coefficients distribution
                    ax2.hist(coef, bins=20, alpha=0.7, color='green', edgecolor='black')
                    ax2.axvline(coef.mean(), color='red', linestyle='--', 
                               label=f'Mean: {coef.mean():.4f}')
                    ax2.axvline(np.median(coef), color='orange', linestyle='--', 
                               label=f'Median: {np.median(coef):.4f}')
                    ax2.set_xlabel('Coefficient Value', fontsize=12)
                    ax2.set_ylabel('Frequency', fontsize=12)
                    ax2.set_title('Coefficients Distribution', fontsize=14, fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    coef_path = f"{plots_dir}/coefficients.png"
                    plt.savefig(coef_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots['coefficients'] = f"plots/{model_name}/coefficients.png"
            except Exception as e:
                print(f"Could not generate coefficients plot: {e}")
        
    except Exception as e:
        print(f"Error generating model summary plots: {e}")
    
    return plots
