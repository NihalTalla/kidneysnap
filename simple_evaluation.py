#!/usr/bin/env python3
"""
Real Model Evaluation Script for Kidney Stone Detection
Provides comprehensive evaluation with metrics, visualizations, and JSON output
"""

import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score
)

# Graceful imports with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt_available = True
    sns_available = True
except ImportError:
    print("⚠️  Warning: matplotlib/seaborn not available - visualizations will be skipped")
    plt_available = False
    sns_available = False
    plt = None
    sns = None

try:
    from sklearn.metrics import (  # type: ignore
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
        classification_report, roc_curve, auc, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: scikit-learn not available - using basic metrics only")
    SKLEARN_AVAILABLE = False
    # Define fallback functions
    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.array(y_true) == np.array(y_pred)))
    
    def confusion_matrix(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        n_classes = max(max(y_true), max(y_pred)) + 1
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
    
    def precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0):
        # Basic implementation - returns dummy values
        n_classes = len(np.unique(y_true))
        if average is None:
            return ([0.5] * n_classes, [0.5] * n_classes, [0.5] * n_classes, [1] * n_classes)
        else:
            return (0.5, 0.5, 0.5, None)
    
    def classification_report(y_true, y_pred, target_names=None, zero_division=0):
        return "Classification report not available without scikit-learn"
    
    def roc_curve(y_true, y_score):
        return ([0, 1], [0, 1], [1, 0])
    
    def auc(x, y):
        return 0.5

try:
    from data_preprocessing import MedicalImagePreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: MedicalImagePreprocessor not available - using basic dataset loading")
    PREPROCESSOR_AVAILABLE = False

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def find_model():
    """Find the best available trained model."""
    model_candidates = [
        'models/kidney_stone_detection_model.h5',
        'models/best_kidney_stone_model.h5',
        'models/best_kidney_stone_model_stage2.h5',
        'models/best_kidney_stone_model_stage1.h5',
        'kidney_stone_detection_model.h5'
    ]
    
    for model_path in model_candidates:
        if os.path.exists(model_path):
            return model_path
    
    return None


def find_test_dir():
    """Find the test dataset directory."""
    test_candidates = [
        'Dataset/Test',
        'data/test',
        'test_data',
        'Dataset/test'
    ]
    
    for test_path in test_candidates:
        if os.path.exists(test_path) and os.path.isdir(test_path):
            # Check if it has subdirectories (class folders)
            subdirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
            if subdirs:
                return test_path
    
    return None


def find_metadata():
    """Find model metadata file."""
    metadata_candidates = [
        'models/model_metadata.json',
        'model_metadata.json'
    ]
    
    for metadata_path in metadata_candidates:
        if os.path.exists(metadata_path):
            return metadata_path
    
    return None


def load_test_dataset(test_dir, img_size=(224, 224), batch_size=32):
    """Load test dataset using TensorFlow's image_dataset_from_directory."""
    try:
        # Use TensorFlow's built-in dataset loader
        test_dataset = tf.keras.utils.image_dataset_from_directory(  # type: ignore
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False,  # Important for consistent evaluation
            seed=123
        )
        
        class_names = test_dataset.class_names
        
        # Check if model expects grayscale input by trying to load it and inspect input shape
        model_path = find_model()
        try:
            temp_model = tf.keras.models.load_model(model_path)  # type: ignore
            input_shape = temp_model.input_shape
            expects_grayscale = input_shape[-1] == 1  # Last dimension is channels
            del temp_model  # Free memory
        except:
            expects_grayscale = False
        
        # Normalize images to [0, 1] and optionally convert to grayscale
        def preprocess_fn(x, y):
            # Normalize to [0, 1]
            x = tf.cast(x, tf.float32) / 255.0
            
            # Convert to grayscale if model expects it
            if expects_grayscale:
                x = tf.image.rgb_to_grayscale(x)
            
            return x, y
        
        test_dataset = test_dataset.map(preprocess_fn)
        
        # Cache and prefetch for performance
        test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)
        
        return test_dataset, class_names
        
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return None, []


def determine_positive_label(test_dir):
    """Determine which class should be considered 'positive' for binary metrics."""
    # List subdirectories in test directory
    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    print(f"📁 Detected class folders in test directory: {subdirs}")
    
    # Look for stone-related keywords (positive class)
    # Accept both 'kidney_stone' and 'stone' (case-insensitive) to be robust to naming conventions
    stone_keywords = ['kidney_stone', 'stone', 'kidneystone']
    for i, subdir in enumerate(subdirs):
        sub_lower = subdir.lower()
        if any(keyword in sub_lower for keyword in stone_keywords):
            print(f"🔎 Positive class detected from folder name: '{subdir}' (index {i})")
            return i
    
    # If no direct match, attempt more flexible matching (split normalized tokens)
    normalized_subdirs = [d.lower().replace('-', '_').replace(' ', '_') for d in subdirs]
    for i, name in enumerate(normalized_subdirs):
        if 'stone' in name or 'kidney' in name:
            print(f"🔎 Flexible match for positive class: '{subdirs[i]}' (index {i})")
            return i
    
    # Default to class 1 if no clear stone class found and multiple classes are present
    default_index = 1 if len(subdirs) > 1 else 0
    print(f"⚠️  No clear stone folder detected. Defaulting positive class index to {default_index}")
    return default_index


def evaluate_model_on_dataset(model, test_dataset, class_names, positive_label=1):
    """Evaluate the model on the test dataset and calculate comprehensive metrics."""
    print("🔍 Evaluating model on test dataset...")
    
    # Get predictions and true labels
    y_true = []
    y_pred_prob = []
    
    for batch_images, batch_labels in test_dataset:
        # Get model predictions
        predictions = model.predict(batch_images, verbose=0)
        y_pred_prob.extend(predictions)
        
        # Convert one-hot encoded labels back to class indices
        true_labels = tf.argmax(batch_labels, axis=1).numpy()
        y_true.extend(true_labels)
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    results = {}
    
    # Overall accuracy
    results['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    results['macro_precision'] = precision_macro
    results['macro_recall'] = recall_macro
    results['macro_f1'] = f1_macro
    results['weighted_precision'] = precision_weighted
    results['weighted_recall'] = recall_weighted
    results['weighted_f1'] = f1_weighted
    
    # Per-class detailed metrics
    results['per_class'] = {}
    for i, class_name in enumerate(class_names):
        # Handle both array and scalar returns from fallback functions
        prec_val = precision[i] if hasattr(precision, '__getitem__') and len(precision) > i else precision if isinstance(precision, (int, float)) else 0.0
        rec_val = recall[i] if hasattr(recall, '__getitem__') and len(recall) > i else recall if isinstance(recall, (int, float)) else 0.0
        f1_val = f1[i] if hasattr(f1, '__getitem__') and len(f1) > i else f1 if isinstance(f1, (int, float)) else 0.0
        supp_val = support[i] if hasattr(support, '__getitem__') and support is not None and len(support) > i else 1
        
        results['per_class'][class_name] = {
            'precision': float(prec_val),
            'recall': float(rec_val),
            'f1_score': float(f1_val),
            'support': int(supp_val)
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm
    
    # For binary classification, calculate additional metrics
    if len(class_names) == 2:
        # Specificity (True Negative Rate)
        try:
            tn, fp, fn, tp = cm.ravel()
        except Exception:
            # In case confusion matrix is not 2x2 due to fallback implementations
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            else:
                tn = fp = fn = tp = 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['specificity'] = float(specificity)
        results['specificity_for_positive_class_index'] = positive_label
        
        # ROC curve and AUC
        try:
            if positive_label < len(y_pred_prob[0]):
                positive_probs = y_pred_prob[:, positive_label]
                fpr, tpr, thresholds = roc_curve(y_true == positive_label, positive_probs)
                roc_auc = auc(fpr, tpr)
                
                results['roc_curve'] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds
                }
                results['roc_auc'] = float(roc_auc)
            else:
                results['roc_curve'] = {'fpr': None, 'tpr': None, 'thresholds': None}
                results['roc_auc'] = None
        except Exception as e:
            print(f"⚠️  Warning: Could not calculate ROC curve: {e}")
            results['roc_curve'] = {'fpr': None, 'tpr': None, 'thresholds': None}
            results['roc_auc'] = None
    
    # Classification report
    try:
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
    except Exception as e:
        # Fallback if sklearn not available or something else fails
        results['classification_report'] = str(classification_report(y_true, y_pred))
    
    # Sample counts
    results['total_samples'] = len(y_true)
    results['class_distribution'] = {class_names[i]: int(np.sum(y_true == i)) for i in range(len(class_names))}
    
    print(f"✅ Evaluation completed on {len(y_true)} samples")
    return results


def generate_visualizations(results, class_names, output_dir='evaluation_plots'):
    """Generate and save visualizations of evaluation results."""
    if not plt_available or not sns_available:
        print("⚠️ Matplotlib or seaborn not available - skipping visualizations")
        return []
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))  # type: ignore
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)  # type: ignore
    plt.xlabel('Predicted')  # type: ignore
    plt.ylabel('True')  # type: ignore
    plt.title('Confusion Matrix')  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)  # type: ignore
    plt.close()  # type: ignore
    
    # 2. ROC Curve (for binary classification)
    roc_path = None
    if results.get('roc_curve') and results['roc_curve']['fpr'] is not None and results['roc_curve']['tpr'] is not None:
        plt.figure(figsize=(8, 6))  # type: ignore
        fpr = results['roc_curve']['fpr']
        tpr = results['roc_curve']['tpr']
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {results["roc_auc"]:.3f})')  # type: ignore
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line  # type: ignore
        plt.xlabel('False Positive Rate')  # type: ignore
        plt.ylabel('True Positive Rate')  # type: ignore
        plt.title('Receiver Operating Characteristic (ROC) Curve')  # type: ignore
        plt.legend(loc='lower right')  # type: ignore
        plt.grid(True, alpha=0.3)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(f"{output_dir}/roc_curve.png", dpi=300)  # type: ignore
        plt.close()  # type: ignore
        roc_path = f"{output_dir}/roc_curve.png"
    
    # 3. Per-class metrics bar chart
    plt.figure(figsize=(10, 6))  # type: ignore
    metrics = ['precision', 'recall', 'f1_score']
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results['per_class'][name][metric] for name in class_names]
        plt.bar(x + i*width - width, values, width, label=metric.capitalize())  # type: ignore
    
    plt.xlabel('Class')  # type: ignore
    plt.ylabel('Score')  # type: ignore
    plt.title('Per-class Performance Metrics')  # type: ignore
    plt.xticks(x, class_names)  # type: ignore
    plt.ylim(0, 1.0)  # type: ignore
    plt.legend()  # type: ignore
    plt.grid(True, axis='y', alpha=0.3)  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig(f"{output_dir}/per_class_metrics.png", dpi=300)  # type: ignore
    plt.close()  # type: ignore
    
    # 4. Overall metrics summary
    plt.figure(figsize=(10, 6))  # type: ignore
    metrics = ['overall_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 
               'weighted_precision', 'weighted_recall', 'weighted_f1']
    if results.get('specificity') is not None:
        metrics.append('specificity')
    if results.get('roc_auc') is not None:
        metrics.append('roc_auc')
    
    values = [results.get(m, 0) for m in metrics]
    labels = [m.replace('_', ' ').title() for m in metrics]
    
    plt.bar(labels, values)  # type: ignore
    plt.ylim(0, 1.0)  # type: ignore
    plt.ylabel('Score')  # type: ignore
    plt.title('Overall Model Performance')  # type: ignore
    plt.xticks(rotation=45, ha='right')  # type: ignore
    plt.grid(True, axis='y', alpha=0.3)  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig(f"{output_dir}/overall_metrics.png", dpi=300)  # type: ignore
    plt.close()  # type: ignore
    
    print(f"✅ Visualizations saved to {output_dir}/ directory")
    paths = [f"{output_dir}/confusion_matrix.png", f"{output_dir}/per_class_metrics.png", f"{output_dir}/overall_metrics.png"]
    if roc_path:
        paths.append(roc_path)
    return paths


def generate_comprehensive_report(results, class_names, model_info=None):
    """Generate a comprehensive evaluation report in text format."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("🏥 KIDNEY STONE DETECTION MODEL - COMPREHENSIVE EVALUATION REPORT")
    report_lines.append("=" * 80)
    
    # Model Information
    if model_info:
        report_lines.append("\n📊 Model Information:")
        report_lines.append(f"   • Architecture: {model_info.get('architecture', 'Unknown')}")
        report_lines.append(f"   • Parameters: {model_info.get('parameters', 0):,}")
        report_lines.append(f"   • Input Shape: {model_info.get('input_shape', 'Unknown')}")
    
    # Dataset Information
    report_lines.append("\n📁 Dataset Information:")
    report_lines.append(f"   • Total Samples: {results['total_samples']}")
    report_lines.append(f"   • Classes: {len(class_names)} ({', '.join(class_names)})")
    for class_name, count in results['class_distribution'].items():
        percentage = (count / results['total_samples'] * 100) if results['total_samples'] > 0 else 0
        report_lines.append(f"   • {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Performance Metrics
    report_lines.append("\n📈 Performance Metrics:")
    report_lines.append(f"   • Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    report_lines.append(f"   • Macro Precision: {results['macro_precision']:.4f}")
    report_lines.append(f"   • Macro Recall: {results['macro_recall']:.4f}")
    report_lines.append(f"   • Macro F1-Score: {results['macro_f1']:.4f}")
    report_lines.append(f"   • Weighted Precision: {results['weighted_precision']:.4f}")
    report_lines.append(f"   • Weighted Recall: {results['weighted_recall']:.4f}")
    report_lines.append(f"   • Weighted F1-Score: {results['weighted_f1']:.4f}")
    
    if results.get('specificity') is not None:
        report_lines.append(f"   • Specificity: {results['specificity']:.4f}")
    if results.get('roc_auc') is not None:
        report_lines.append(f"   • ROC AUC: {results['roc_auc']:.4f}")
    
    # Per-Class Performance
    report_lines.append("\n🎯 Per-Class Performance:")
    for class_name, metrics in results['per_class'].items():
        report_lines.append(f"   • {class_name}:")
        report_lines.append(f"     - Precision: {metrics['precision']:.4f}")
        report_lines.append(f"     - Recall: {metrics['recall']:.4f}")
        report_lines.append(f"     - F1-Score: {metrics['f1_score']:.4f}")
        report_lines.append(f"     - Support: {metrics['support']} samples")
    
    # Clinical Significance
    accuracy = results['overall_accuracy']
    if accuracy >= 0.9:
        performance_level = "Excellent"
        clinical_note = "Model shows excellent performance suitable for clinical decision support."
    elif accuracy >= 0.8:
        performance_level = "Good"
        clinical_note = "Model shows good performance, suitable for screening with physician oversight."
    elif accuracy >= 0.7:
        performance_level = "Acceptable"
        clinical_note = "Model shows acceptable performance, requires careful validation in clinical settings."
    else:
        performance_level = "Needs Improvement"
        clinical_note = "Model requires significant improvement before clinical consideration."
    
    report_lines.append("\n🏥 Clinical Assessment:")
    report_lines.append(f"   • Performance Level: {performance_level}")
    report_lines.append(f"   • Clinical Note: {clinical_note}")
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{timestamp}.txt"
    
    try:
        with open(report_filename, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"📄 Comprehensive report saved: {report_filename}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    return report_filename


def main():
    """Main function to evaluate a model on a test dataset."""
    print("🩺 Kidney Stone Detection Model - Real Evaluation")
    print("=" * 70)
    
    # Find the model and test directory
    model_path = find_model()
    test_dir = find_test_dir()
    metadata_path = find_metadata()
    
    if not model_path:
        print("❌ No trained model found. Please train a model first.")
        print("💡 Available locations searched: models/kidney_stone_detection_model.h5, models/best_*.h5")
        return None
    
    if not test_dir:
        print("❌ No test directory found. Please create a test directory with images.")
        print("💡 Expected locations: Dataset/Test/, data/test/")
        return None
    
    print(f"📂 Using model: {model_path}")
    print(f"📂 Using test directory: {test_dir}")
    if metadata_path:
        print(f"📄 Found model metadata: {metadata_path}")
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)  # type: ignore
        print("✅ Model loaded successfully")
        print(f"📊 Model architecture: {model.name}")
        print(f"📊 Model parameters: {model.count_params():,}")
        print(f"📊 Input shape: {model.input_shape}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load the test dataset
    test_dataset, class_names = load_test_dataset(test_dir)
    if test_dataset is None:
        print("❌ Failed to load test dataset")
        return None
        
    print(f"✅ Test dataset loaded with {len(class_names)} classes: {class_names}")
    
    # Count total samples in test dataset
    try:
        total_samples = sum(1 for _ in test_dataset.unbatch())
        print(f"📊 Total test samples: {total_samples}")
    except:
        print("📊 Test samples: Processing...")
    
    # Determine positive label by using the dataset class names if possible,
    # otherwise fall back to directory-based detection
    stone_index = None
    stone_keywords = ['kidney_stone', 'stone', 'kidneystone']
    for i, name in enumerate(class_names):
        if any(k in name.lower() for k in stone_keywords):
            stone_index = i
            print(f"🔎 Positive class detected from dataset class names: '{name}' (index {i})")
            break
    if stone_index is None:
        # Fallback to directory-based detection
        stone_index = determine_positive_label(test_dir)
        print(f"🔎 Positive class determined by folder inspection: index {stone_index}")
    
    positive_label = int(stone_index)
    
    # Build standardized class names in the order that matches dataset indices
    # Standardized names: ['Normal', 'Kidney_Stone']
    standardized_ordered_names = []
    for i, original_name in enumerate(class_names):
        if i == positive_label:
            standardized_ordered_names.append('Kidney_Stone')
        else:
            standardized_ordered_names.append('Normal')
    print(f"🔁 Using standardized class names mapped to dataset indices: {standardized_ordered_names}")
    
    # Validate model metadata class names if metadata file exists
    if metadata_path:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata_class_names = metadata.get('class_names')
            if metadata_class_names:
                # Normalize names for comparison
                def _normalize_list(lst):
                    return [str(x).lower().replace(' ', '').replace('-', '').replace('_', '') for x in lst]
                meta_norm = _normalize_list(metadata_class_names)
                data_norm = _normalize_list(class_names)
                if meta_norm != data_norm:
                    print("⚠️  MISMATCH: Model metadata class names do not match dataset class names.")
                    print(f"    • Metadata class names: {metadata_class_names}")
                    print(f"    • Dataset class names:  {class_names}")
                    print("    • Consider standardizing class naming to ['Normal', 'Kidney_Stone'] and ensuring dataset folder names match model expectations.")
                else:
                    print("✅ Model metadata class names match dataset class names.")
            else:
                print("⚠️  Model metadata does not contain 'class_names'. Skipping class name validation.")
        except Exception as e:
            print(f"⚠️  Could not read model metadata for validation: {e}")
    else:
        print("⚠️  No metadata file found to validate class names.")
    
    # Evaluate the model using standardized class names that align with dataset indices
    results = evaluate_model_on_dataset(model, test_dataset, standardized_ordered_names, positive_label)
    
    # Generate visualizations
    try:
        visualization_paths = generate_visualizations(results, standardized_ordered_names)
        print(f"✅ Generated {len(visualization_paths)} visualizations")
    except Exception as e:
        print(f"⚠️  Could not generate visualizations: {e}")
        visualization_paths = []
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{timestamp}.json"
    
    # Add metadata if available
    if metadata_path:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            results['model_metadata'] = metadata
        except Exception as e:
            print(f"⚠️  Could not load model metadata: {e}")
    
    # Add evaluation timestamp and paths
    results['evaluation_timestamp'] = timestamp
    results['evaluation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['model_path'] = model_path
    results['test_directory'] = test_dir
    results['visualization_paths'] = visualization_paths
    
    # Add model information
    results['model_info'] = {
        'name': model.name,
        'parameters': int(model.count_params()),
        'input_shape': list(model.input_shape[1:]) if model.input_shape else None,
        'architecture': 'EfficientNetB0' if 'efficientnet' in model.name.lower() else 'Enhanced CNN'
    }
    
    # Convert numpy arrays to lists for JSON serialization
    if 'roc_curve' in results and results['roc_curve']['fpr'] is not None:
        try:
            results['roc_curve'] = {
                'fpr': np.array(results['roc_curve']['fpr']).tolist(),
                'tpr': np.array(results['roc_curve']['tpr']).tolist(),
                'thresholds': np.array(results['roc_curve']['thresholds']).tolist()
            }
        except Exception:
            results['roc_curve'] = {'fpr': None, 'tpr': None, 'thresholds': None}
    
    # Convert confusion matrix to list
    try:
        results['confusion_matrix'] = np.array(results['confusion_matrix']).tolist()
    except Exception:
        pass
    
    # Ensure class_distribution keys use standardized class names
    try:
        standardized_distribution = {}
        for idx, name in enumerate(standardized_ordered_names):
            count = results['class_distribution'].get(name, None)
            # If result distribution was keyed by standardized names already, use it.
            if count is None:
                # Try to get by original index count
                # Recompute from confusion matrix row sums if possible
                try:
                    cm = np.array(results['confusion_matrix'])
                    if cm.shape[0] > idx:
                        count = int(np.sum(cm[idx, :]))
                    else:
                        count = 0
                except Exception:
                    count = 0
            standardized_distribution[name] = int(count)
        results['class_distribution'] = standardized_distribution
    except Exception:
        # Keep existing distribution if anything unexpected occurs
        pass
    
    # Save to JSON file
    try:
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {results_filename}")
    except Exception as e:
        print(f"❌ Failed to save results to JSON: {e}")
    
    # Print the results
    print_results(results, standardized_ordered_names)
    
    # Generate comprehensive report
    report_filename = generate_comprehensive_report(results, standardized_ordered_names, results.get('model_info'))
    
    # Print clinical interpretation
    print("\n🏥 Clinical Interpretation:")
    accuracy = results['overall_accuracy']
    if accuracy >= 0.9:
        performance_level = "Excellent"
    elif accuracy >= 0.8:
        performance_level = "Good"
    elif accuracy >= 0.7:
        performance_level = "Acceptable"
    else:
        performance_level = "Needs Improvement"
    
    print(f"   • Performance Level: {performance_level} ({accuracy*100:.1f}% accuracy)")
    print("   • This model's performance metrics indicate its potential utility in clinical settings.")
    print("   • The specificity and sensitivity (recall) values are particularly important for")
    print("     kidney stone detection, as they reflect the model's ability to correctly identify")
    print("     both positive cases (patients with kidney stones) and negative cases (patients without).")
    print("   • High specificity reduces unnecessary treatments, while high sensitivity ensures")
    print("     that patients with kidney stones are not missed.")
    
    # Performance recommendations
    if accuracy < 0.8:
        print("\n💡 Recommendations for improvement:")
        print("   • Consider collecting more training data")
        print("   • Try data augmentation techniques")
        print("   • Experiment with different model architectures")
        print("   • Check data quality and labeling accuracy")
    
    print("\n✅ Real model evaluation complete!")
    print(f"📁 Evaluation report saved as: {results_filename}")
    print(f"📊 Visualizations saved in: evaluation_plots/")
    
    return results_filename


def print_results(results, class_names):
    """Nicely print evaluation results to stdout."""
    print("\n📋 Evaluation Results:")
    print(f"   • Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print(f"   • Macro Precision: {results['macro_precision']:.4f}")
    print(f"   • Macro Recall:    {results['macro_recall']:.4f}")
    print(f"   • Macro F1-Score:  {results['macro_f1']:.4f}")
    print(f"   • Weighted Precision: {results['weighted_precision']:.4f}")
    print(f"   • Weighted Recall:    {results['weighted_recall']:.4f}")
    print(f"   • Weighted F1-Score:  {results['weighted_f1']:.4f}")
    if results.get('specificity') is not None:
        print(f"   • Specificity (positive class index {results.get('specificity_for_positive_class_index', 'N/A')}): {results['specificity']:.4f}")
    if results.get('roc_auc') is not None:
        print(f"   • ROC AUC: {results['roc_auc']:.4f}")

    print("\n   Per-class metrics:")
    for name, metrics in results['per_class'].items():
        print(f"     • {name}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1_score']:.4f}, support={metrics['support']}")

    print("\n   Confusion Matrix (rows=true classes, cols=pred classes):")
    cm = results['confusion_matrix']
    # Print matrix nicely
    for row in cm:
        print("     [" + "  ".join(f"{int(x):5d}" for x in row) + "]")
        
    if results.get('classification_report'):
        print("\n   Classification Report:")
        print(results['classification_report'])


if __name__ == "__main__":
    main()