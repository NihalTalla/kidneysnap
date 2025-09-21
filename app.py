from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessing
model = None
class_names = ['Normal', 'Kidney_Stone']

def load_model():
    """Load the trained kidney stone detection model"""
    global model
    try:
        # Try to load the model
        model_path = 'kidney_stone_model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)  # type: ignore
            logger.info(f"✅ Model loaded successfully from {model_path}")
            logger.info(f"📊 Model input shape: {model.input_shape}")
            return True
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_image(image_data, target_size=(224, 224)):
    """
    Dynamic preprocessing for model prediction with automatic channel detection
    Supports both RGB and grayscale models with automatic adaptation
    
    Args:
        image_data: PIL Image or numpy array
        target_size: Target size for the model
    
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = image_data
        
        # Determine model input requirements
        model_input_shape = model.input_shape if model else None
        expected_channels = model_input_shape[-1] if model_input_shape else 3
        
        logger.info(f"Model expects {expected_channels} channels, input shape: {model_input_shape}")
        
        # Convert image based on model requirements
        if expected_channels == 1:
            # Model expects grayscale
            if image.mode != 'L':
                image = image.convert('L')  # Convert to grayscale
        else:
            # Model expects RGB (3 channels)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        
        # Handle channel dimension
        if expected_channels == 1 and len(img_array.shape) == 2:
            # Add channel dimension for grayscale
            img_array = np.expand_dims(img_array, axis=-1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Preprocessed image shape: {img_array.shape}")
        return img_array
    
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def analyze_image_features(image_array):
    """
    Direct image analysis for kidney stone detection with high confidence threshold
    
    Args:
        image_array: Preprocessed image array
    
    Returns:
        dict: Analysis results with confidence score
    """
    try:
        # Convert back to uint8 for analysis
        img_uint8 = (image_array[0] * 255).astype(np.uint8)
        
        # Handle both RGB and grayscale inputs for analysis
        if len(img_uint8.shape) == 3 and img_uint8.shape[-1] == 3:
            # RGB image - convert to grayscale
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        elif len(img_uint8.shape) == 3 and img_uint8.shape[-1] == 1:
            # Grayscale with channel dimension - squeeze
            gray = np.squeeze(img_uint8, axis=-1)
        else:
            # Already grayscale 2D
            gray = img_uint8
        
        # Enhanced stone detection features
        features = {}
        
        # 1. High contrast circular/oval objects (potential stones)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        stone_indicators = 0
        confidence = 0.0
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            stone_indicators += len(circles) * 0.3
            features['circular_objects'] = len(circles)
        else:
            features['circular_objects'] = 0
        
        # 2. High intensity regions (calcified stones appear bright)
        bright_threshold = np.percentile(gray, 85)
        bright_pixels = np.sum(gray > bright_threshold)
        bright_ratio = bright_pixels / (gray.shape[0] * gray.shape[1])
        
        if bright_ratio > 0.1:  # More than 10% bright pixels
            stone_indicators += bright_ratio * 2
        
        features['bright_pixel_ratio'] = bright_ratio
        
        # 3. Edge density (stones have distinct edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        if edge_density > 0.05:
            stone_indicators += edge_density * 1.5
        
        features['edge_density'] = edge_density
        
        # 4. Texture analysis using standard deviation
        texture_std = np.std(gray)
        if texture_std > 30:  # High texture variation
            stone_indicators += (texture_std / 100) * 1.2
        
        features['texture_variation'] = texture_std
        
        # Calculate confidence with high precision threshold (0.8 as per specification)
        confidence = min(float(stone_indicators / 3.0), 1.0)  # Normalize to 0-1
        
        # Apply high precision threshold - only trust image analysis if confidence > 0.8
        is_reliable = confidence > 0.8
        
        return {
            'confidence': confidence,
            'is_reliable': is_reliable,
            'features': features,
            'stone_indicators': stone_indicators
        }
    
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}")
        return {
            'confidence': 0.0,
            'is_reliable': False,
            'features': {},
            'stone_indicators': 0
        }

def predict_kidney_stone(image_data):
    """
    Comprehensive kidney stone prediction with high precision focus
    
    Args:
        image_data: PIL Image
    
    Returns:
        dict: Prediction results with confidence and explanation
    """
    try:
        if model is None:
            return {
                'error': 'Model not loaded',
                'prediction': 'Unknown',
                'confidence': 0.0
            }

        # Preprocess image
        processed_image = preprocess_image(image_data)

        # Get model prediction
        model_prediction = model.predict(processed_image, verbose=0)
        model_confidence = float(np.max(model_prediction))
        model_class_idx = int(np.argmax(model_prediction))
        model_class = class_names[model_class_idx]

        # Get direct image analysis
        image_analysis = analyze_image_features(processed_image)

        # Integration logic with high precision focus
        final_prediction = model_class
        final_confidence = model_confidence
        explanation = f"Model prediction: {model_class} ({model_confidence:.3f})"

        # Only trust image analysis if confidence > 0.8 (high threshold as per specification)
        if image_analysis['is_reliable']:
            image_suggests_stone = image_analysis['confidence'] > 0.5

            if image_suggests_stone and model_class == 'Kidney_Stone':
                # Both methods agree on stone - cautious boosting
                final_confidence = min(model_confidence * 1.1, 0.99)
                explanation += f" | Image analysis confirms stone presence ({image_analysis['confidence']:.3f})"

            elif image_suggests_stone and model_class == 'Normal':
                # Conflict - prioritize precision, lean towards stone if image analysis is very confident
                if image_analysis['confidence'] > 0.85:
                    final_prediction = 'Kidney_Stone'
                    final_confidence = image_analysis['confidence']
                    explanation += f" | Image analysis overrides: strong stone indicators ({image_analysis['confidence']:.3f})"
                else:
                    explanation += f" | Image analysis suggests stone but model disagrees - keeping model prediction"

            else:
                explanation += f" | Image analysis supports model prediction"

        else:
            explanation += f" | Image analysis inconclusive ({image_analysis['confidence']:.3f})"

        # Apply conservative threshold for final stone detection
        if final_prediction == 'Kidney_Stone' and final_confidence < 0.6:
            final_prediction = 'Uncertain'
            explanation += " | Confidence too low for definitive stone detection"

        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'model_prediction': model_class,
            'model_confidence': model_confidence,
            'image_analysis': image_analysis,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {
            'error': str(e),
            'prediction': 'Error',
            'confidence': 0.0
        }

@app.route('/')
def index():
    """Serve the main upload interface"""
    return send_from_directory('.', 'upload_interface.html')

@app.route('/evaluation')
def evaluation():
    """Serve the evaluation preview"""
    return send_from_directory('.', 'evaluation_preview.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and kidney stone prediction
    
    Expected input: 
    - 'image' file in multipart/form-data
    OR
    - JSON with base64 encoded image
    
    Returns:
    - JSON with prediction results
    """
    try:
        image_data = None
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read and validate image
            try:
                image_data = Image.open(io.BytesIO(file.read()))
            except Exception as e:
                return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Handle JSON with base64 image
        elif request.is_json:
            json_data = request.get_json()
            if 'image' in json_data:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(json_data['image'])
                    image_data = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
        
        if image_data is None:
            return jsonify({'error': 'No image provided'}), 400
        
        # Make prediction
        result = predict_kidney_stone(image_data)
        
        # Log prediction
        logger.info(f"Prediction made: {result.get('prediction', 'Unknown')} "
                   f"(confidence: {result.get('confidence', 0):.3f})")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    return jsonify({
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'parameters': model.count_params(),
        'class_names': class_names
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("🚀 Starting Flask server with kidney stone detection model")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to load model. Please ensure kidney_stone_model.h5 exists.")
        print("\n⚠️  To use the prediction service, you need to:")
        print("1. Train a model using your training script")
        print("2. Save it as 'kidney_stone_model.h5'")
        print("3. Restart this server")
        
        # Start server anyway for evaluation viewing
        logger.info("🚀 Starting Flask server in evaluation-only mode")
        app.run(debug=True, host='0.0.0.0', port=5000)