"""
Model Evaluation Script for Rockburst Prediction
================================================
This module provides comprehensive evaluation and analysis of the trained
Random Forest model including performance metrics, visualizations, and reports.

Author: Data Science Team GAMMA
Created: August 10, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Add project root to path
sys.path.append('.')
sys.path.append('./models')

from model import RockburstRandomForestModel
from exception_logging.logger import get_logger


class ModelEvaluator:
    """
    Comprehensive evaluation of the rockburst prediction model.
    """
    
    def __init__(self, model_dir='./artifacts/models', output_dir='./artifacts/evaluation'):
        """
        Initialize the evaluator.
        
        Args:
            model_dir: Directory containing the trained model
            output_dir: Directory to save evaluation results
        """
        self.logger = get_logger("model_evaluator")
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.model = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_model(self):
        """
        Load the trained model for evaluation.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            self.model = RockburstRandomForestModel()
            self.model.load_model(self.model_dir)
            self.logger.info("✅ Model loaded successfully for evaluation")
            return True
        except (FileNotFoundError, Exception) as e:
            self.logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def generate_evaluation_data(self, n_samples=1000):
        """
        Generate evaluation data with known ground truth.
        
        Args:
            n_samples: Number of evaluation samples
            
        Returns:
            pd.DataFrame: Evaluation data
        """
        self.logger.info(f"🎲 Generating {n_samples} evaluation data points...")
        
        np.random.seed(456)  # Different seed for evaluation
        
        # Generate the 9 user input features
        data = {
            'Duration_days': np.random.uniform(1, 30, n_samples),
            'Energy_Unit_log': np.random.uniform(2, 8, n_samples),
            'Energy_density_Joule_sqr': np.random.uniform(10, 1000, n_samples),
            'Volume_m3_sqr': np.random.uniform(5, 500, n_samples),
            'Event_freq_unit_per_day_log': np.random.uniform(0, 5, n_samples),
            'Energy_Joule_per_day_sqr': np.random.uniform(20, 2000, n_samples),
            'Volume_m3_per_day_sqr': np.random.uniform(10, 800, n_samples),
            'Energy_per_Volume_log': np.random.uniform(0, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target
        risk_score = (
            0.25 * (df['Energy_density_Joule_sqr'] / df['Energy_density_Joule_sqr'].max()) +
            0.20 * (df['Event_freq_unit_per_day_log'] / df['Event_freq_unit_per_day_log'].max()) +
            0.25 * (df['Energy_per_Volume_log'] / df['Energy_per_Volume_log'].max()) +
            0.15 * (df['Energy_Joule_per_day_sqr'] / df['Energy_Joule_per_day_sqr'].max()) +
            0.15 * (df['Volume_m3_per_day_sqr'] / df['Volume_m3_per_day_sqr'].max())
        )
        
        risk_score += np.random.normal(0, 0.1, n_samples)
        risk_score = np.clip(risk_score, 0, 1)
        
        intensity_labels = []
        for score in risk_score:
            if score <= 0.35:
                intensity_labels.append(0)  # Low
            elif score <= 0.70:
                intensity_labels.append(1)  # Medium
            else:
                intensity_labels.append(2)  # High
                
        df['Intensity_Level_encoded'] = intensity_labels
        
        self.logger.info("✅ Evaluation data generated successfully")
        return df
    
    def calculate_performance_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_names = ['Low', 'Medium', 'High']
        class_report = classification_report(y_true, y_pred, 
                                           target_names=class_names,
                                           output_dict=True)
        
        # ROC AUC (multiclass)
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = None
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_auc_multiclass': float(roc_auc) if roc_auc is not None else None,
            'class_names': class_names
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Rockburst Prediction')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, feature_names, importances, top_n=20, save_path=None):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        # Get top N features
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Feature importance plot saved to: {save_path}")
            
        plt.close()
    
    def plot_class_distribution(self, y_true, y_pred, class_names, save_path=None):
        """
        Plot class distribution comparison.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            class_names: List of class names
            save_path: Path to save plot
        """
        true_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        
        # Ensure all classes are represented
        for i in range(len(class_names)):
            if i not in true_counts:
                true_counts[i] = 0
            if i not in pred_counts:
                pred_counts[i] = 0
        
        true_counts = true_counts.sort_index()
        pred_counts = pred_counts.sort_index()
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, true_counts.values, width, label='True', alpha=0.8)
        plt.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.8)
        
        plt.xlabel('Rockburst Intensity Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution: True vs Predicted')
        plt.xticks(x, class_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Class distribution plot saved to: {save_path}")
            
        plt.close()
    
    def analyze_prediction_confidence(self, y_pred_proba):
        """
        Analyze prediction confidence statistics.
        
        Args:
            y_pred_proba: Prediction probabilities
            
        Returns:
            dict: Confidence analysis
        """
        max_probs = np.max(y_pred_proba, axis=1)
        
        confidence_stats = {
            'mean_confidence': float(np.mean(max_probs)),
            'median_confidence': float(np.median(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs)),
            'confidence_quartiles': np.percentile(max_probs, [25, 50, 75]).tolist(),
            'low_confidence_samples': int(np.sum(max_probs < 0.6)),
            'high_confidence_samples': int(np.sum(max_probs > 0.8))
        }
        
        return confidence_stats
    
    def generate_evaluation_report(self, eval_data, target_column='Intensity_Level_encoded'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            eval_data: Evaluation dataset
            target_column: Name of target column
            
        Returns:
            dict: Comprehensive evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.logger.info("📊 Generating comprehensive evaluation report...")
        
        # Prepare data
        X_eval, y_true = self.model.prepare_features(eval_data, target_column)
        
        # Make predictions
        y_pred = self.model.predict(X_eval)
        y_pred_proba = self.model.predict_proba(X_eval)
        
        # Calculate metrics
        performance_metrics = self.calculate_performance_metrics(y_true, y_pred, y_pred_proba)
        
        # Analyze confidence
        confidence_analysis = self.analyze_prediction_confidence(y_pred_proba)
        
        # Get feature importance
        feature_importance = None
        if hasattr(self.model.model, 'feature_importances_'):
            feature_importance = {
                'feature_names': list(X_eval.columns),
                'importances': self.model.model.feature_importances_.tolist()
            }
        
        # Create evaluation report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.model.get_model_info(),
            'evaluation_data': {
                'n_samples': len(eval_data),
                'n_features': X_eval.shape[1],
                'class_distribution': pd.Series(y_true).value_counts().sort_index().to_dict()
            },
            'performance_metrics': performance_metrics,
            'confidence_analysis': confidence_analysis,
            'feature_importance': feature_importance
        }
        
        # Generate plots
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion matrix plot
        cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(performance_metrics['confusion_matrix'], 
                                 performance_metrics['class_names'], cm_path)
        
        # Class distribution plot
        dist_path = os.path.join(plots_dir, 'class_distribution.png')
        self.plot_class_distribution(y_true, y_pred, 
                                   performance_metrics['class_names'], dist_path)
        
        # Feature importance plot
        if feature_importance:
            importance_path = os.path.join(plots_dir, 'feature_importance.png')
            self.plot_feature_importance(feature_importance['feature_names'],
                                       feature_importance['importances'],
                                       save_path=importance_path)
        
        # Save evaluation report
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        self.logger.info(f"📄 Evaluation report saved to: {report_path}")
        
        return evaluation_report
    
    def print_evaluation_summary(self, evaluation_report):
        """
        Print a formatted evaluation summary.
        
        Args:
            evaluation_report: Evaluation report dictionary
        """
        print("="*80)
        print("📊 MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Basic info
        model_info = evaluation_report['model_info']
        eval_data = evaluation_report['evaluation_data']
        metrics = evaluation_report['performance_metrics']
        confidence = evaluation_report['confidence_analysis']
        
        print(f"🕒 Evaluation timestamp: {evaluation_report['timestamp']}")
        print(f"📊 Evaluation samples: {eval_data['n_samples']:,}")
        print(f"🔧 Features used: {eval_data['n_features']:,}")
        print(f"🤖 Model trained: {model_info['is_trained']}")
        
        print(f"\\n📈 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision_weighted']:.4f}")
        print(f"   Recall:    {metrics['recall_weighted']:.4f}")
        print(f"   F1-Score:  {metrics['f1_weighted']:.4f}")
        
        print(f"\\n🎯 PER-CLASS PERFORMANCE:")
        class_names = metrics['class_names']
        for i, class_name in enumerate(class_names):
            print(f"   {class_name:7}: P={metrics['precision_per_class'][i]:.3f} "
                  f"R={metrics['recall_per_class'][i]:.3f} "
                  f"F1={metrics['f1_per_class'][i]:.3f}")
        
        print(f"\\n🔍 CONFIDENCE ANALYSIS:")
        print(f"   Mean confidence: {confidence['mean_confidence']:.3f}")
        print(f"   Median confidence: {confidence['median_confidence']:.3f}")
        print(f"   Low confidence samples (<0.6): {confidence['low_confidence_samples']}")
        print(f"   High confidence samples (>0.8): {confidence['high_confidence_samples']}")
        
        print(f"\\n📊 CLASS DISTRIBUTION:")
        for class_idx, count in eval_data['class_distribution'].items():
            class_name = class_names[class_idx]
            percentage = (count / eval_data['n_samples']) * 100
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        if evaluation_report['feature_importance']:
            print(f"\\n🔝 TOP 5 IMPORTANT FEATURES:")
            features = evaluation_report['feature_importance']['feature_names']
            importances = evaluation_report['feature_importance']['importances']
            top_features = sorted(zip(features, importances), 
                                key=lambda x: x[1], reverse=True)[:5]
            for i, (feature, importance) in enumerate(top_features):
                print(f"   {i+1}. {feature}: {importance:.4f}")
        
        print("="*80)


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Random Forest model for rockburst prediction')
    parser.add_argument('--model-dir', type=str, default='./artifacts/models',
                       help='Directory containing the trained model')
    parser.add_argument('--eval-data', type=str,
                       help='Path to evaluation data CSV file (optional)')
    parser.add_argument('--output-dir', type=str, default='./artifacts/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of evaluation samples to generate (if no eval data file)')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model_dir, args.output_dir)
        
        print("📊 Rockburst Model Evaluation Script")
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Load model
        if not evaluator.load_model():
            raise RuntimeError("Failed to load model for evaluation")
        
        # Load or generate evaluation data
        if args.eval_data and os.path.exists(args.eval_data):
            print(f"📄 Loading evaluation data from: {args.eval_data}")
            eval_data = pd.read_csv(args.eval_data)
        else:
            print(f"🎲 Generating {args.n_samples} evaluation samples...")
            eval_data = evaluator.generate_evaluation_data(args.n_samples)
        
        # Generate evaluation report
        evaluation_report = evaluator.generate_evaluation_report(eval_data)
        
        # Print summary
        evaluator.print_evaluation_summary(evaluation_report)
        
        print(f"\\n📁 Evaluation results saved to: {args.output_dir}")
        print("   📄 evaluation_report.json - Detailed metrics")
        print("   📊 plots/ - Visualization plots")
        print("="*60)
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
