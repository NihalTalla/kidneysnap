import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Graceful imports with fallback
try:
    from models.capsule_network import create_capsule_model
    CAPSULE_AVAILABLE = True
except ImportError:
    print("Warning: Capsule network not available")
    CAPSULE_AVAILABLE = False

try:
    from data_preprocessing import MedicalImagePreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    print("Warning: MedicalImagePreprocessor not available")
    PREPROCESSOR_AVAILABLE = False

def evaluate_model(model_path=None, test_data_dir='data/test'):
    """Evaluate the trained model or create a demo evaluation"""
    
    # Check if model exists
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        model_type = "Trained Model"
    else:
        print("No trained model found.")
        if CAPSULE_AVAILABLE:
            print("Creating new model for architecture demonstration...")
            model = create_capsule_model()
            model_type = "Untrained Model (Demo)"
        else:
            print("Capsule network not available. Using demo evaluation only.")
            return create_demo_evaluation(None, "Demo Model (No Network)")
    
    # Check if test data exists
    if not os.path.exists(test_data_dir):
        print(f"Test data directory '{test_data_dir}' not found.")
        print("Creating synthetic evaluation data for demonstration...")
        return create_demo_evaluation(model, model_type)
    
    print(f"Loading test data from {test_data_dir}...")
    
    # Load test data
    if PREPROCESSOR_AVAILABLE:
        preprocessor = MedicalImagePreprocessor()
        
        try:
            test_gen, _ = preprocessor.load_dataset(test_data_dir, batch_size=32)
            
            if test_gen.samples == 0:
                print("No test samples found. Creating demo evaluation...")
                return create_demo_evaluation(model, model_type)
                
        except Exception as e:
            print(f"Error loading test data: {e}")
            print("Creating demo evaluation...")
            return create_demo_evaluation(model, model_type)
    else:
        print("Data preprocessor not available. Creating demo evaluation...")
        return create_demo_evaluation(model, model_type)
    
    print(f"Found {test_gen.samples} test samples")
    
    # Get predictions
    try:
        predictions = model.predict(test_gen)
        
        # Handle different model output formats
        if isinstance(predictions, tuple):
            class_pred, detection_pred = predictions
        else:
            class_pred = predictions
            detection_pred = None
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return create_demo_evaluation(model, model_type)
    
    # Get true labels
    y_true = test_gen.classes
    y_pred = np.argmax(class_pred, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    
    # Classification report
    class_names = ['Normal', 'Kidney Stone']
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC curve
    if class_pred.shape[1] > 1:
        fpr, tpr, _ = roc_curve(y_true, class_pred[:, 1])
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
    
    # Create evaluation plots
    create_evaluation_plots(cm, fpr, tpr, roc_auc, class_names, accuracy, model_type)
    
    # Print results
    print(f"\n=== {model_type} Evaluation Results ===")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'model_type': model_type
    }

def create_demo_evaluation(model, model_type):
    """Create a demo evaluation with synthetic data"""
    print("\n=== Creating Demo Evaluation ===")
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100
    n_normal = 60
    n_stones = 40
    
    # Create synthetic predictions (simulating model behavior)
    if "Trained" in model_type:
        # Simulate better performance for "trained" model
        normal_pred = np.random.beta(8, 2, n_normal)  # Higher confidence for normal
        stone_pred = np.random.beta(2, 8, n_stones)   # Higher confidence for stones
    else:
        # Simulate random performance for untrained model
        normal_pred = np.random.beta(5, 5, n_normal)  # Random predictions
        stone_pred = np.random.beta(5, 5, n_stones)   # Random predictions
    
    # Create true labels
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_stones)])
    
    # Create class probabilities
    class_probs = np.column_stack([
        np.concatenate([normal_pred, 1-stone_pred]),
        np.concatenate([1-normal_pred, stone_pred])
    ])
    
    # Get predictions
    y_pred = np.argmax(class_probs, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    
    # Classification report
    class_names = ['Normal', 'Kidney Stone']
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, class_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Create evaluation plots
    create_evaluation_plots(cm, fpr, tpr, roc_auc, class_names, accuracy, f"{model_type} (Demo Data)")
    
    # Print results
    print(f"\n=== {model_type} Demo Evaluation Results ===")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"AUC Score: {roc_auc:.3f}")
    print("\nClassification Report:")
    print(report)
    print("\n⚠️  Note: This is a demonstration using synthetic data.")
    print("   For real evaluation, train the model and provide test data.")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'model_type': f"{model_type} (Demo)",
        'is_demo': True
    }

def create_evaluation_plots(cm, fpr, tpr, roc_auc, class_names, accuracy, title):
    """Create and save evaluation plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - Evaluation Results', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    
    # Accuracy by class
    if cm.sum(axis=1).min() > 0:  # Avoid division by zero
        class_acc = cm.diagonal() / cm.sum(axis=1)
        bars = axes[1, 0].bar(class_names, class_acc, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('Accuracy by Class')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_acc):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.2%}', ha='center', va='bottom')
    
    # Overall Performance Summary
    performance_text = f"""Overall Accuracy: {accuracy:.2%}
AUC Score: {roc_auc:.3f}

Model Architecture:
• Capsule Network
• Input: 224×224×3
• Classes: {len(class_names)}"""
    
    axes[1, 1].text(0.05, 0.95, performance_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Performance Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'model_evaluation_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Evaluation plots saved as '{filename}'")
    plt.show()

if __name__ == "__main__":
    print("🩺 Kidney Stone Detection Model Evaluation")
    print("=" * 50)
    
    # Try to evaluate with trained model first, fallback to demo
    results = evaluate_model('kidney_stone_capsule_model.h5', 'data/test')
    
    print("\n✅ Evaluation completed!")
    if results.get('is_demo', False):
        print("\n💡 To run real evaluation:")
        print("   1. Train the model using train_capsule_model.py")
        print("   2. Add test images to data/test/ directory")
        print("   3. Run this script again")
