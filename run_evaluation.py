#!/usr/bin/env python3
"""
Quick Evaluation Runner - Shows the system in action
"""

import os
import sys

def run_evaluation():
    """Run the evaluation and show results"""
    print("🩺 KIDNEY STONE DETECTION - EVALUATION RUNNER")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("simple_evaluation.py"):
        print("❌ simple_evaluation.py not found")
        return
    
    # Check for model and data
    model_exists = os.path.exists("models/kidney_stone_detection_model.h5")
    data_exists = os.path.exists("Dataset/Test")
    
    print(f"📁 Model available: {'✅' if model_exists else '❌'}")
    print(f"📁 Test data available: {'✅' if data_exists else '❌'}")
    
    if not model_exists or not data_exists:
        print("\n⚠️ Missing required files for evaluation")
        return
    
    print("\n🔄 Starting comprehensive model evaluation...")
    print("This will take a few moments...")
    
    try:
        # Import the evaluation main function
        from simple_evaluation import main as evaluation_main
        
        # Run the evaluation
        result_file = evaluation_main()
        
        if result_file:
            print(f"\n✅ SUCCESS! Evaluation completed")
            print(f"📊 Results file: {result_file}")
            
            # Show what was generated
            if os.path.exists("evaluation_plots"):
                plots = os.listdir("evaluation_plots")
                print(f"📊 Generated {len(plots)} visualization files")
            
            return True
        else:
            print("⚠️ Evaluation completed with issues")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False

if __name__ == "__main__":
    success = run_evaluation()
    if success:
        print("\n🎉 Evaluation system is working perfectly!")
    else:
        print("\n⚠️ Check the error messages above")