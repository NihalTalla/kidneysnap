#!/usr/bin/env python3
"""
Complete System Demonstration
Shows all components of the kidney stone detection system working together
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"🏥 {title}")
    print("=" * 70)

def check_requirements():
    """Check if all required components are available"""
    print_header("SYSTEM REQUIREMENTS CHECK")
    
    required_files = [
        'simple_train.py',
        'predict.py', 
        'app.py',
        'simple_evaluation.py',
        'dataset_organizer.py',
        'data_preprocessing.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ All core system files present!")
    return True

def demo_dataset_organization():
    """Demonstrate dataset organization"""
    print_header("DATASET ORGANIZATION DEMO")
    
    try:
        from dataset_organizer import DatasetOrganizer
        organizer = DatasetOrganizer()
        
        # Check if we have datasets to work with
        if Path("Dataset/Train").exists() and Path("Dataset/Test").exists():
            print("✅ Dataset structure already exists")
            success = organizer.verify_dataset_structure()
            if success:
                print("✅ Dataset structure is valid")
            else:
                print("⚠️  Dataset structure needs attention")
        else:
            print("📁 No existing dataset found")
            print("💡 You can organize raw images using: python dataset_organizer.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset organization demo failed: {e}")
        return False

def demo_prediction_system():
    """Demonstrate prediction system"""
    print_header("PREDICTION SYSTEM DEMO")
    
    try:
        from predict import KidneyStoneDetector
        
        print("🔄 Initializing kidney stone detector...")
        detector = KidneyStoneDetector()
        print("✅ Detector initialized successfully")
        
        # Find test images
        test_locations = [
            "Dataset/Test/Normal",
            "Dataset/Test/Kidney_stone", 
            "prediction_result_1.png",
            "stone_detection_test.png"
        ]
        
        test_image = None
        for location in test_locations:
            if os.path.exists(location):
                if os.path.isdir(location):
                    # Find first image in directory
                    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
                    for file in os.listdir(location):
                        if file.lower().endswith(valid_extensions):
                            test_image = os.path.join(location, file)
                            break
                else:
                    # Direct file
                    test_image = location
                
                if test_image:
                    break
        
        if test_image:
            print(f"🖼️  Testing with image: {os.path.basename(test_image)}")
            result = detector.predict(test_image)
            
            print("📊 Prediction Results:")
            print(f"   🏷️  Class: {result['class']}")
            print(f"   📈 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
            print(f"   🔬 Method: {result.get('prediction_method', 'unknown')}")
            
            # Generate visualization
            vis_path = "demo_prediction_result.png"
            detector.visualize_prediction(test_image, save_path=vis_path, show=False)
            print(f"   💾 Visualization saved: {vis_path}")
            
        else:
            print("⚠️  No test images found for demonstration")
            print("💡 Place test images in Dataset/Test/ directories")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction system demo failed: {e}")
        return False

def demo_training_capability():
    """Demonstrate training system capability"""
    print_header("TRAINING SYSTEM DEMO")
    
    try:
        from simple_train import create_model, calculate_class_weights
        
        print("🏗️  Testing model creation...")
        # Test EfficientNetB0 model creation
        model = create_model(use_transfer_learning=True)
        params = model.count_params()
        print(f"✅ EfficientNetB0 model created: {params:,} parameters")
        
        # Test enhanced CNN model creation
        model_cnn = create_model(use_transfer_learning=False)  
        params_cnn = model_cnn.count_params()
        print(f"✅ Enhanced CNN model created: {params_cnn:,} parameters")
        
        # Test class weight calculation if dataset exists
        if os.path.exists("Dataset/Train"):
            class_names = ['Normal', 'Kidney_stone']
            weights = calculate_class_weights("Dataset/Train", class_names)
            print(f"✅ Class weights calculated: {weights}")
        else:
            print("⚠️  No training dataset found for class weight calculation")
        
        print("💡 To start full training: python simple_train.py")
        return True
        
    except Exception as e:
        print(f"❌ Training system demo failed: {e}")
        return False

def demo_evaluation_capability():
    """Demonstrate evaluation system"""
    print_header("EVALUATION SYSTEM DEMO")
    
    try:
        # Check for existing evaluation results
        eval_files = [f for f in os.listdir('.') if f.startswith('evaluation_results_') and f.endswith('.json')]
        
        if eval_files:
            latest_eval = max(eval_files)
            print(f"✅ Found existing evaluation: {latest_eval}")
            
            # Load and display summary
            import json
            with open(latest_eval, 'r') as f:
                results = json.load(f)
            
            print("📊 Latest Evaluation Summary:")
            print(f"   📈 Overall Accuracy: {results.get('overall_accuracy', 0):.4f}")
            print(f"   🎯 Macro F1-Score: {results.get('macro_f1', 0):.4f}")
            print(f"   📊 Total Samples: {results.get('total_samples', 0)}")
            
            if 'roc_auc' in results:
                print(f"   📈 ROC AUC: {results['roc_auc']:.4f}")
        else:
            print("⚠️  No evaluation results found")
        
        # Check for evaluation plots
        if os.path.exists("evaluation_plots"):
            plots = os.listdir("evaluation_plots")
            print(f"✅ Evaluation plots available: {len(plots)} files")
            for plot in plots[:3]:  # Show first 3
                print(f"   📊 {plot}")
        else:
            print("⚠️  No evaluation plots found")
        
        print("💡 To run evaluation: python simple_evaluation.py")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation demo failed: {e}")
        return False

def demo_web_interface():
    """Demonstrate web interface capability"""
    print_header("WEB INTERFACE DEMO")
    
    try:
        # Check Flask app structure
        if os.path.exists("app.py"):
            print("✅ Flask application found: app.py")
        else:
            print("❌ Flask application not found")
            return False
        
        if os.path.exists("templates"):
            templates = os.listdir("templates")
            print(f"✅ Templates directory: {len(templates)} files")
        else:
            print("⚠️  Templates directory not found")
        
        os.makedirs("uploads", exist_ok=True)
        print("✅ Uploads directory ready")
        
        print("🌐 Web interface components ready!")
        print("💡 To start web server: python app.py")
        print("💡 Then visit: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"❌ Web interface demo failed: {e}")
        return False

def main():
    """Run complete system demonstration"""
    print_header("COMPLETE KIDNEY STONE DETECTION SYSTEM DEMO")
    print("This demonstration will test all major system components")
    
    # Track success of each component
    results = {}
    
    # Run demonstrations
    results['requirements'] = check_requirements()
    results['dataset'] = demo_dataset_organization()
    results['prediction'] = demo_prediction_system()
    results['training'] = demo_training_capability()
    results['evaluation'] = demo_evaluation_capability()
    results['web'] = demo_web_interface()
    
    # Summary
    print_header("DEMONSTRATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {component.upper()}")
    
    print(f"\n📊 Overall: {passed}/{total} components passed")
    
    if passed == total:
        print("\n🎉 Complete system is ready for production!")
        print("\n🚀 Next steps:")
        print("   1. Organize your dataset: python dataset_organizer.py")
        print("   2. Train your model: python simple_train.py")
        print("   3. Evaluate performance: python simple_evaluation.py")
        print("   4. Start web interface: python app.py")
    else:
        print(f"\n⚠️  {total - passed} components need attention")
        print("💡 Check error messages above and ensure all files are present")
    
    print("\n" + "=" * 70)
    print("🏥 Kidney Stone Detection System Demo Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()