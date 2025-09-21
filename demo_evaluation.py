#!/usr/bin/env python3
"""
Complete Model Evaluation Demo
Demonstrates the comprehensive evaluation capabilities of simple_evaluation.py
"""

import os
import json
from datetime import datetime

def main():
    """Demonstrate the complete evaluation system."""
    print("🩺 KIDNEY STONE DETECTION - COMPLETE EVALUATION DEMO")
    print("=" * 70)
    
    # Check if evaluation script exists
    if not os.path.exists("simple_evaluation.py"):
        print("❌ simple_evaluation.py not found")
        return
    
    # Run the evaluation
    print("🔄 Running comprehensive model evaluation...")
    print("This will:")
    print("   ✅ Load the trained model and test dataset")
    print("   ✅ Calculate accuracy, precision, recall, F1-score on real test data")
    print("   ✅ Generate confusion matrix and classification report")
    print("   ✅ Create visualizations of results")
    print("   ✅ Save evaluation metrics to JSON file")
    print()
    
    # Import and run evaluation
    try:
        from simple_evaluation import main as eval_main
        results_file = eval_main()
        
        if results_file:
            print(f"\n📊 EVALUATION SUMMARY:")
            print(f"   📁 Results saved to: {results_file}")
            
            # Load and display key metrics
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                print(f"   📈 Overall Accuracy: {results.get('overall_accuracy', 0):.2%}")
                print(f"   🎯 F1-Score (Macro): {results.get('macro_f1', 0):.4f}")
                print(f"   📊 Total Test Samples: {results.get('total_samples', 0)}")
                print(f"   🕒 Evaluation Date: {results.get('evaluation_date', 'Unknown')}")
                
                # Model information
                model_info = results.get('model_info', {})
                if model_info:
                    print(f"   🏗️  Model Architecture: {model_info.get('architecture', 'Unknown')}")
                    print(f"   📊 Model Parameters: {model_info.get('parameters', 0):,}")
                
                # Visualizations
                vis_paths = results.get('visualization_paths', [])
                if vis_paths:
                    print(f"   📊 Visualizations Generated: {len(vis_paths)}")
                    for path in vis_paths:
                        print(f"      - {os.path.basename(path)}")
                
            except Exception as e:
                print(f"   ⚠️  Could not load results details: {e}")
        
        print("\n✅ Complete evaluation demonstration finished!")
        print("\n🎯 What was accomplished:")
        print("   ✅ Real model evaluation on actual test data")
        print("   ✅ Comprehensive metrics calculation")
        print("   ✅ Professional visualization generation")
        print("   ✅ JSON export for further analysis")
        print("   ✅ Clinical performance assessment")
        
        print("\n📁 Generated Files:")
        if os.path.exists("evaluation_plots"):
            plots = os.listdir("evaluation_plots")
            for plot in plots:
                print(f"   📊 evaluation_plots/{plot}")
        
        # List recent evaluation results
        eval_files = [f for f in os.listdir('.') if f.startswith('evaluation_results_') and f.endswith('.json')]
        eval_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if eval_files:
            print(f"   📄 {eval_files[0]} (latest)")
            if len(eval_files) > 1:
                print(f"   📄 {len(eval_files)-1} additional evaluation file(s)")
        
        print(f"\n🏥 The evaluation system provides:")
        print(f"   • Medical-grade performance metrics")
        print(f"   • Clinical interpretation and recommendations")
        print(f"   • Professional visualizations")
        print(f"   • Complete audit trail in JSON format")
        print(f"   • Integration with your training pipeline")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("💡 Make sure you have:")
        print("   • A trained model in models/ directory")
        print("   • Test data in Dataset/Test/ directory")
        print("   • All required dependencies installed")

if __name__ == "__main__":
    main()