#!/usr/bin/env python3
"""
Convert trained Keras models to TensorFlow Lite format for Android deployment
Supports various optimization strategies for mobile deployment
"""

import tensorflow as tf
import numpy as np
import os
import argparse
from pathlib import Path

def load_representative_dataset(data_dir="Dataset/Test", num_samples=100):
    """
    Load a representative dataset for quantization calibration
    
    Args:
        data_dir: Directory containing test images
        num_samples: Number of samples to use for calibration
    
    Returns:
        Generator yielding representative data samples
    """
    def representative_data_gen():
        # Load test dataset for calibration
        if os.path.exists(data_dir):
            test_dataset = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                image_size=(128, 128),
                batch_size=1,
                label_mode=None,  # We only need images for calibration
                shuffle=True,
                seed=42
            )
            
            count = 0
            for image_batch in test_dataset:
                if count >= num_samples:
                    break
                
                # Normalize image to [0, 1] range (same as training)
                image = tf.cast(image_batch, tf.float32) / 255.0
                yield [image]
                count += 1
        else:
            print(f"Warning: Test data directory {data_dir} not found. Using random data for calibration.")
            # Fallback to random data if test directory doesn't exist
            for _ in range(num_samples):
                yield [np.random.random((1, 128, 128, 3)).astype(np.float32)]
    
    return representative_data_gen

def convert_to_tflite(model_path, output_path, optimization_type="default"):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to the saved Keras model (.h5 file)
        output_path: Path where the .tflite file will be saved
        optimization_type: Type of optimization to apply
                          - "default": Basic optimization
                          - "dynamic": Dynamic range quantization
                          - "int8": Full integer quantization
                          - "float16": Float16 quantization
    
    Returns:
        Path to the converted .tflite file
    """
    
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimization based on type
    if optimization_type == "default":
        print("🔧 Applying default optimization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    elif optimization_type == "dynamic":
        print("🔧 Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    elif optimization_type == "int8":
        print("🔧 Applying full integer quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = load_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
    elif optimization_type == "float16":
        print("🔧 Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    else:
        print("⚠️ Unknown optimization type, using default...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    try:
        print("🔄 Converting model to TensorFlow Lite...")
        tflite_model = converter.convert()
        print("✅ Model converted successfully")
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None
    
    # Save the converted model
    try:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"💾 TensorFlow Lite model saved to: {output_path}")
        
        # Print model size information
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = original_size / tflite_size if tflite_size > 0 else 0
        
        print(f"📊 Original model size: {original_size:.2f} MB")
        print(f"📊 TensorFlow Lite model size: {tflite_size:.2f} MB")
        print(f"📊 Compression ratio: {compression_ratio:.2f}x")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error saving converted model: {e}")
        return None

def test_tflite_model(tflite_path, test_image_path=None):
    """
    Test the converted TensorFlow Lite model
    
    Args:
        tflite_path: Path to the .tflite model
        test_image_path: Optional path to a test image
    """
    
    print(f"\n🧪 Testing TensorFlow Lite model: {tflite_path}")
    
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ Model loaded successfully in TensorFlow Lite interpreter")
        print(f"📋 Input shape: {input_details[0]['shape']}")
        print(f"📋 Input type: {input_details[0]['dtype']}")
        print(f"📋 Output shape: {output_details[0]['shape']}")
        print(f"📋 Output type: {output_details[0]['dtype']}")
        
        # Test with random input if no test image provided
        if test_image_path and os.path.exists(test_image_path):
            # Load and preprocess test image
            image = tf.keras.utils.load_img(test_image_path, target_size=(128, 128))
            image_array = tf.keras.utils.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array.astype(np.float32) / 255.0
            print(f"🖼️ Using test image: {test_image_path}")
        else:
            # Use random input for testing
            image_array = np.random.random((1, 128, 128, 3)).astype(np.float32)
            print("🎲 Using random input for testing")
        
        # Handle different input types for quantized models
        if input_details[0]['dtype'] == np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Handle different output types
        if output_details[0]['dtype'] == np.uint8:
            # Dequantize if needed
            scale, zero_point = output_details[0]['quantization']
            if scale != 0:
                output_data = scale * (output_data - zero_point)
        
        print(f"🎯 Model output: {output_data}")
        
        # Interpret results for binary classification
        if output_data.shape[-1] == 2:
            class_names = ['Kidney_Stone', 'Normal']
            predicted_class = np.argmax(output_data[0])
            confidence = np.max(output_data[0])
            print(f"🏷️ Predicted class: {class_names[predicted_class]}")
            print(f"📊 Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing TensorFlow Lite model: {e}")
        return False

def main():
    """Main function to handle command line arguments and conversion"""
    
    parser = argparse.ArgumentParser(description='Convert Keras model to TensorFlow Lite')
    parser.add_argument('--model', '-m', type=str, default='kidney_stone_model_best.h5',
                       help='Path to the Keras model file (.h5)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the .tflite file')
    parser.add_argument('--optimization', '-opt', type=str, default='dynamic',
                       choices=['default', 'dynamic', 'int8', 'float16'],
                       help='Optimization type for the model')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test the converted model after conversion')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Path to test image for model testing')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Available model files:")
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                print(f"  - {file}")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        model_name = Path(args.model).stem
        args.output = f"{model_name}_{args.optimization}.tflite"
    
    print("=" * 60)
    print("KERAS TO TENSORFLOW LITE CONVERTER")
    print("=" * 60)
    print(f"📁 Input model: {args.model}")
    print(f"📁 Output file: {args.output}")
    print(f"🔧 Optimization: {args.optimization}")
    print("=" * 60)
    
    # Convert the model
    tflite_path = convert_to_tflite(args.model, args.output, args.optimization)
    
    if tflite_path and args.test:
        # Test the converted model
        test_tflite_model(tflite_path, args.test_image)
    
    print("\n" + "=" * 60)
    if tflite_path:
        print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"📱 Your model is ready for Android deployment: {tflite_path}")
    else:
        print("❌ CONVERSION FAILED!")
    print("=" * 60)

def convert_all_optimizations(model_path):
    """
    Convert model with all optimization types for comparison
    
    Args:
        model_path: Path to the Keras model file
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    model_name = Path(model_path).stem
    optimizations = ['default', 'dynamic', 'int8', 'float16']
    
    print("🔄 Converting model with all optimization types...")
    
    results = {}
    for opt in optimizations:
        output_path = f"{model_name}_{opt}.tflite"
        print(f"\n--- Converting with {opt} optimization ---")
        
        tflite_path = convert_to_tflite(model_path, output_path, opt)
        if tflite_path:
            size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
            results[opt] = {'path': tflite_path, 'size': size}
            
            # Quick test
            if test_tflite_model(tflite_path):
                results[opt]['test_passed'] = True
            else:
                results[opt]['test_passed'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    for opt, info in results.items():
        status = "✅" if info['test_passed'] else "❌"
        print(f"{status} {opt:10} | {info['size']:6.2f} MB | {info['path']}")
    print("=" * 60)

if __name__ == "__main__":
    # Check if we want to convert all optimizations
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--all":
        model_path = os.sys.argv[2] if len(os.sys.argv) > 2 else "kidney_stone_model_best.h5"
        convert_all_optimizations(model_path)
    else:
        main()