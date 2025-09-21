# Kidney Stone Binary Classifier (Simple CNN, 128x128, TFLite)

This repository contains a focused, minimal pipeline for training a lightweight convolutional neural network (CNN) to perform binary classification of kidney images:
- Normal Kidney
- Kidney with Stones

The project is optimized for mobile deployment on Android using TensorFlow Lite (.tflite). Input images are 128x128 RGB. The target is a practical accuracy of at least 85% while keeping the model small and fast.

Contents
- simple_train.py        — Training script for a small CNN (128x128 input)
- simple_evaluation.py   — Evaluation script (accuracy, precision, recall, F1, confusion matrix)
- predict.py             — Single-image prediction script (128x128 input)
- data_preprocessing.py  — Image resize/normalization helpers for training and inference
- setup_data.py          — Optional helper to prepare/resize dataset to 128x128
- model_to_tflite.py     — Convert trained Keras (.h5) models to TensorFlow Lite (.tflite) with quantization options
- requirements.txt       — Minimal required packages
- data/                  — Expected dataset folder structure (see below)

Why this project
- Simple and maintainable codebase for binary classification
- Input images standardized to 128x128 RGB for mobile efficiency
- Lightweight model suitable for Android deployment with TensorFlow Lite
- Clear instructions for training, evaluation, conversion, and inference

Dataset requirements
- Recommended total images: ~2000 (adjustable)
- Balanced classes are important for stable training
- Suggested split: 70% train / 15% validation / 15% test

Expected directory structure:
data/
├── train/
│   ├── Normal/
│   └── Kidney_Stone/
├── validation/
│   ├── Normal/
│   └── Kidney_Stone/
└── test/
    ├── Normal/
    └── Kidney_Stone/

Note: Folder names are case-sensitive on some systems. This project expects the class folders to be named "Normal" and "Kidney_Stone".

Getting started

1. Install dependencies
- The repository is intentionally minimal. Install required packages with:
pip install -r requirements.txt

Minimum required packages include:
- tensorflow
- numpy
- opencv-python (cv2)
- pillow
- scikit-learn
- matplotlib
- pandas

2. Prepare your dataset
- Organize images into the folder structure above.
- Ensure images are RGB or convertible to RGB.
- Images will be resized to 128x128 by data_preprocessing.py or setup_data.py.

3. (Optional) Run dataset preparation
- If you need to resize and organize images, use:
python setup_data.py --source <raw_images_folder> --output data/ --img_size 128

4. Train the model
- Train a lightweight CNN designed for 128x128 inputs:
python simple_train.py --data_dir data/ --img_size 128 --batch_size 32 --epochs 30 --output_model model.h5

- Tips to reach ~85% accuracy:
  - Ensure class balance or use class weights.
  - Use data augmentation (rotation, flips, brightness).
  - Start with a small model and increase capacity if underfitting.
  - Try transfer learning (MobileNetV2 backbone) if accuracy is low while keeping it lightweight.

5. Evaluate the model
- Evaluate on the held-out test set with:
python simple_evaluation.py --model model.h5 --test_dir data/test/ --img_size 128

- The evaluation script reports:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix
  - Per-class support counts

6. Convert to TensorFlow Lite (for Android)
- Convert the trained Keras model (.h5) to a .tflite file suitable for Android:
python model_to_tflite.py --keras_model model.h5 --output_model model.tflite

- Available conversion options:
  - Default float32 TFLite
  - Post-training dynamic range quantization
  - Post-training full integer quantization (representative dataset required for best results)

- Example for dynamic range quantization:
python model_to_tflite.py --keras_model model.h5 --output_model model_dynamic.tflite --quantize dynamic

- For full integer quantization (smaller and faster on many devices), provide a small representative dataset via the script's options.

7. Inference (Python example)
- Use predict.py to run a single image inference:
python predict.py --model model.h5 --image sample.jpg --img_size 128

- Output:
  - Predicted label: "Normal" or "Kidney_Stone"
  - Confidence score (probability)

8. Android integration (high-level notes)
- Place the generated .tflite model in your Android project's assets folder.
- Use TensorFlow Lite Android support libraries (org.tensorflow:tensorflow-lite and, if using NNAPI or GPU delegates, the appropriate delegate packages).
- Preprocess camera/image frames to 128x128 RGB and normalize exactly as in data_preprocessing.py before feeding to the model.
- Postprocess model output to threshold probabilities and return a friendly label and confidence.

Best practices and tips
- Keep the input preprocessing (resize, normalize) identical between training and production.
- Test the .tflite model on a few representative device samples to verify real-world accuracy.
- If model size or latency is critical, prefer quantized models (dynamic or integer).
- Monitor class imbalance—accuracy alone can be misleading when classes are imbalanced; prefer F1-score and per-class recall.

Troubleshooting
- If training accuracy is high but validation/test accuracy is low:
  - Reduce overfitting: add dropout, augment data, or use L2 regularization.
  - Collect more data or use transfer learning.
- If model is too slow on device:
  - Convert to quantized TFLite.
  - Use a smaller architecture (fewer layers/filters) or MobileNet variants.
- If predictions are inconsistent:
  - Verify preprocessing pipeline matches between training, evaluation, and inference.
  - Ensure images are in RGB and have the same scale/range (e.g., 0-1 float or 0-255 int).

License & disclaimer
- License: MIT (or choose an appropriate license and add LICENSE file)
- Medical disclaimer: This model is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

Contributing
- Keep changes simple and focused on the training/inference pipeline.
- Raise issues for bugs or feature requests.
- Pull requests should include tests or validation results where applicable.

Contact
- For questions about training, evaluation, or TFLite conversion, open an issue in the repository with details about your dataset and the problem you're encountering.

This README focuses on a practical, minimal pipeline for binary kidney stone classification and Android deployment. For advanced clinical workflows, multi-modal imaging, or object detection tasks, additional domain expertise and regulatory considerations are required.