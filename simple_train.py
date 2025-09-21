#!/usr/bin/env python3
"""
Simple and effective kidney stone detection model training
Uses your actual dataset: Dataset/Train and Dataset/Test
"""

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from typing import Tuple, Dict, List

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model(use_transfer_learning=True) -> tf.keras.Model:
    """Create an enhanced CNN model with transfer learning option

    Args:
        use_transfer_learning: If True, uses EfficientNetB0 as base model

    Returns:
        Compiled (not compiled here) tf.keras.Model. If transfer learning is used,
        the returned model has attribute `base_model` pointing to the EfficientNetB0 instance.
    """
    if use_transfer_learning:
        # Use EfficientNetB0 as base model (smaller than ResNet50, better for mobile)
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',  # Pre-trained on ImageNet
            include_top=False,   # Exclude classification layer
            input_shape=(224, 224, 3)
        )

        # Freeze the base model layers initially (will be unfrozen during fine-tuning)
        base_model.trainable = False

        # Create new model on top
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))

        # Preprocessing specific to EfficientNet
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)

        # Base model
        x = base_model(x, training=False)

        # Combine global average and global max pooling
        gap = tf.keras.layers.GlobalAveragePooling2D()(x)
        gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Concatenate()([gap, gmp])

        # Add batchnorm and augmentation robustness
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Dense layer with L1 regularization for sparsity
        x = tf.keras.layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l1(0.0001)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Smaller dense as intermediate
        x = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l1(0.0001)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Final classification layer
        outputs = tf.keras.layers.Dense(2, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_kidney_cnn")

        # Attach base_model so we can unfreeze parts of it later
        model.base_model = base_model

    else:
        # Enhanced version of the original CNN with more layers and better regularization
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))

        # First block
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Second block
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Third block
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Fourth block
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Global pooling and classification
        gap = tf.keras.layers.GlobalAveragePooling2D()(x)
        gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Concatenate()([gap, gmp])

        x = tf.keras.layers.Dense(512, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(256, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        outputs = tf.keras.layers.Dense(2, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="enhanced_kidney_cnn")

    return model


def calculate_class_weights(train_dir: str, class_names: List[str]) -> Dict[int, float]:
    """
    Calculate class weights based on counts of images in training directory.
    Returns a dictionary {class_index: weight}
    """
    counts = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    for cls in class_names:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            counts.append(0)
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_ext)]
        counts.append(len(files))

    total = sum(counts) if sum(counts) > 0 else 1
    n_classes = len(counts)
    class_weights = {}
    for i, c in enumerate(counts):
        # Inverse frequency: total / (n_classes * count)
        if c > 0:
            class_weights[i] = total / (n_classes * c)
        else:
            class_weights[i] = 1.0
    return class_weights


def safe_random_brightness_layer(factor: float = 0.2):
    """
    Returns a brightness augmentation layer. If RandomBrightness is unavailable in this TF version,
    fall back to a Lambda layer using tf.image.stateless_random_brightness.
    """
    if hasattr(tf.keras.layers, "RandomBrightness"):
        return tf.keras.layers.RandomBrightness(factor)
    else:
        # Stateless random brightness using seed derived from tensor; use a small deterministic op
        def _rand_brightness(x):
            # Use tf.image.random_brightness which is stateful; it's acceptable fallback
            return tf.image.random_brightness(x, max_delta=factor)
        return tf.keras.layers.Lambda(lambda x: _rand_brightness(x))


def create_datasets(train_dir: str,
                    test_dir: str,
                    batch_size: int = 32,
                    img_size: Tuple[int, int] = (224, 224),
                    validation_split: float = 0.2,
                    seed: int = 42):
    """Create training, validation and test datasets with enhanced augmentation

    Args:
        train_dir: Directory containing training images (class subfolders)
        test_dir: Directory containing test images (class subfolders)
        batch_size: Batch size for training
        img_size: Image size for model input
        validation_split: Fraction of training data to use for validation (stratified)
        seed: Random seed used for splitting
    """
    print(f"Loading training data from: {train_dir}")
    print(f"Loading test data from: {test_dir}")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Use image_dataset_from_directory with validation_split for stratified splitting
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='training'
    )

    val_ds = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='validation'
    )

    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False,
        seed=seed
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Enhanced data augmentation using tf.keras.Sequential for more complex augmentations
    # Include RandomTranslation, RandomHeight, RandomWidth, RandomFlip, RandomRotation, RandomZoom,
    # RandomContrast, RandomBrightness(or fallback), and GaussianNoise
    augmentation_layers = [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ]

    # Add RandomHeight and RandomWidth if available
    if hasattr(tf.keras.layers, "RandomHeight"):
        augmentation_layers.append(tf.keras.layers.RandomHeight(0.1))
    if hasattr(tf.keras.layers, "RandomWidth"):
        augmentation_layers.append(tf.keras.layers.RandomWidth(0.1))

    augmentation_layers += [
        tf.keras.layers.RandomContrast(0.15),
        safe_random_brightness_layer(0.1),
        tf.keras.layers.GaussianNoise(0.01),
    ]

    data_augmentation = tf.keras.Sequential(augmentation_layers, name="data_augmentation")

    def preprocess_train(image, label):
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # Apply augmentation
        image = data_augmentation(image)
        return image, label

    def preprocess_val_test(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_dataset = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_ds.map(preprocess_val_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_ds.map(preprocess_val_test, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize performance
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Print dataset sizes in samples (approx) for clarity
    try:
        train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        val_batches = tf.data.experimental.cardinality(val_dataset).numpy()
        test_batches = tf.data.experimental.cardinality(test_dataset).numpy()
        print(f"Training batches: {train_batches}, Validation batches: {val_batches}, Test batches: {test_batches}")
    except Exception:
        pass

    return train_dataset, val_dataset, test_dataset, class_names


def plot_training_history(history, filename='training_history.png'):
    """Plot training history and save to file"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    if 'accuracy' in history.history:
        axes[0, 0].plot(history.history.get('accuracy', []), label='Training', linewidth=2)
        axes[0, 0].plot(history.history.get('val_accuracy', []), label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].axis('off')

    # Loss
    axes[0, 1].plot(history.history.get('loss', []), label='Training', linewidth=2)
    axes[0, 1].plot(history.history.get('val_loss', []), label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history.get('precision', []), label='Training', linewidth=2)
        axes[1, 0].plot(history.history.get('val_precision', []), label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')

    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history.get('recall', []), label='Training', linewidth=2)
        axes[1, 1].plot(history.history.get('val_recall', []), label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    """Custom callback to compute per-class accuracy on validation dataset each epoch."""

    def __init__(self, val_dataset, class_names: List[str]):
        super().__init__()
        self.val_dataset = val_dataset
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        # Accumulate predictions and labels
        y_true = []
        y_pred = []
        for x_batch, y_batch in self.val_dataset:
            preds = self.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1).tolist())
            y_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        per_class_acc = {}
        for i, cls in enumerate(self.class_names):
            cls_idx = i
            mask = (y_true == cls_idx)
            if mask.sum() == 0:
                acc = None
            else:
                acc = (y_pred[mask] == y_true[mask]).mean()
            per_class_acc[cls] = acc

        # Log per-class accuracy to console and logs
        if logs is None:
            logs = {}
        for k, v in per_class_acc.items():
            key = f'val_acc_{k}'
            logs[key] = v if v is not None else 0.0
            print(f"Epoch {epoch+1} - per-class val acc [{k}]: {v if v is not None else 'N/A'}")


def train_model():
    """Main training function with enhanced model and training process"""
    # Configuration
    TRAIN_DIR = "Dataset/Train"
    TEST_DIR = "Dataset/Test"
    BATCH_SIZE = 32  # Increased batch size
    INITIAL_EPOCHS = 20  # Stage 1
    FINE_TUNE_EPOCHS = 30  # Stage 2
    STAGE1_LR = 0.001
    STAGE2_LR = 1e-4
    VALIDATION_SPLIT = 0.2
    LABEL_SMOOTHING = 0.1
    USE_TRANSFER_LEARNING = True  # Use EfficientNetB0 by default
    VALIDATION_ACC_THRESHOLD = 0.80  # required to proceed to stage 2

    print("=" * 60)
    print("KIDNEY STONE DETECTION MODEL TRAINING")
    print("=" * 60)

    # Check directories
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        return
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory not found: {TEST_DIR}")
        return

    # Create datasets with validation split (stratified)
    train_dataset, val_dataset, test_dataset, class_names = create_datasets(
        TRAIN_DIR, TEST_DIR, BATCH_SIZE, img_size=(224, 224), validation_split=VALIDATION_SPLIT
    )

    # Calculate class weights based on folder counts
    class_weights = calculate_class_weights(TRAIN_DIR, class_names)
    print(f"Class weights: {class_weights}")

    # Create model
    print("\nCreating model...")
    model = create_model(use_transfer_learning=USE_TRANSFER_LEARNING)

    # Prepare loss with label smoothing
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    # Stage 1: Frozen Transfer Learning
    print("\nSTAGE 1: Frozen transfer learning (train head only)")
    optimizer_stage1 = Adam(learning_rate=STAGE1_LR)

    model.compile(
        optimizer=optimizer_stage1,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # Model summary
    print("\nModel Summary:")
    model.summary()

    # Ensure models and logs directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Callbacks for stage 1
    stage1_checkpoint = ModelCheckpoint(
        'models/best_kidney_stone_model_stage1.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    stage1_earlystop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    stage1_reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="logs/fit_stage1_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1
    )
    per_class_cb = PerClassAccuracyCallback(val_dataset, class_names)

    callbacks_stage1 = [stage1_earlystop, stage1_reduce_lr, stage1_checkpoint, tb_callback, per_class_cb]

    # Steps per epoch for scheduling
    try:
        steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
        if steps_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY:
            steps_per_epoch = None
    except Exception:
        steps_per_epoch = None

    print(f"\nStarting Stage 1 training for up to {INITIAL_EPOCHS} epochs...")
    history_stage1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks_stage1,
        class_weight=class_weights,
        verbose=1
    )

    # Save Stage 1 history plot
    plot_training_history(history_stage1, filename='training_history_stage1.png')

    # Evaluate Stage 1 on validation to decide if fine-tuning should proceed
    val_metrics = model.evaluate(val_dataset, verbose=0)
    names = model.metrics_names
    val_metrics_dict = dict(zip(names, val_metrics))
    val_acc = val_metrics_dict.get('accuracy', val_metrics_dict.get('categorical_accuracy', 0))
    print(f"\nStage 1 best validation accuracy: {val_acc:.4f}")

    if val_acc < VALIDATION_ACC_THRESHOLD:
        print(f"Validation accuracy ({val_acc:.4f}) did not reach threshold ({VALIDATION_ACC_THRESHOLD}).")
        print("Skipping Stage 2 fine-tuning to avoid overfitting or wasted compute.")
        # Save final model from stage1 as best
        final_model_filename = 'models/kidney_stone_detection_model.h5'
        model.save(final_model_filename)
        # Evaluate on test set and save metadata
        eval_results = model.evaluate(test_dataset, verbose=0)
        metrics_names = model.metrics_names
        metrics_dict = dict(zip(metrics_names, eval_results))
        precision = metrics_dict.get('precision', 0)
        recall = metrics_dict.get('recall', 0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        model_metadata = {
            "model_name": model.name,
            "accuracy": float(metrics_dict.get('accuracy', 0)),
            "precision": float(metrics_dict.get('precision', 0)),
            "recall": float(metrics_dict.get('recall', 0)),
            "f1_score": float(f1),
            "auc": float(metrics_dict.get('auc', 0)),
            "image_size": [224, 224],
            "class_names": class_names,
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs_trained_stage1": len(history_stage1.history.get('loss', [])),
            "batch_size": BATCH_SIZE,
            "transfer_learning": USE_TRANSFER_LEARNING
        }
        with open('models/model_metadata.json', 'w') as f:
            import json
            json.dump(model_metadata, f, indent=4)

        print("\n" + "=" * 60)
        print("✅ MODEL TRAINING COMPLETED (Stage 1 only).")
        print(f"📁 Model saved as: {final_model_filename}")
        print("📁 Best stage1 model saved as: models/best_kidney_stone_model_stage1.h5")
        print("📁 Model metadata saved as: models/model_metadata.json")
        print("📊 Training history saved as: training_history_stage1.png")
        print("=" * 60)
        return model, history_stage1, model_metadata

    # Stage 2: Gradual fine-tuning
    print("\nSTAGE 2: Gradual fine-tuning of top layers of the base model")
    # Unfreeze the top 20% of base_model layers if available
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        total_layers = len(base_model.layers)
        # compute top 20% (at least 1)
        top_pct = 0.20
        num_to_unfreeze = max(1, int(total_layers * top_pct))
        # Unfreeze last num_to_unfreeze layers
        for layer in base_model.layers[-num_to_unfreeze:]:
            layer.trainable = True
        # Keep the rest frozen
        for layer in base_model.layers[:-num_to_unfreeze]:
            layer.trainable = False
        print(f"Unfroze the top {num_to_unfreeze} layers out of {total_layers} in base_model for fine-tuning.")
    else:
        print("No base_model attribute found; skipping selective unfreeze and will unfreeze entire model.")
        for layer in model.layers:
            layer.trainable = True

    # Recompile with lower learning rate and cosine decay schedule
    # Compute steps for scheduling
    try:
        steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
        if steps_per_epoch is None or steps_per_epoch <= 0:
            steps_per_epoch = 100
    except Exception:
        steps_per_epoch = 100

    total_fine_tune_steps = FINE_TUNE_EPOCHS * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=STAGE2_LR,
        decay_steps=max(1, total_fine_tune_steps)
    )
    optimizer_stage2 = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer_stage2,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # Callbacks for stage 2
    stage2_checkpoint = ModelCheckpoint(
        'models/best_kidney_stone_model_stage2.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    stage2_earlystop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    stage2_reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=7,
        min_lr=1e-8,
        verbose=1
    )
    tb_callback2 = tf.keras.callbacks.TensorBoard(
        log_dir="logs/fit_stage2_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1
    )
    per_class_cb2 = PerClassAccuracyCallback(val_dataset, class_names)

    callbacks_stage2 = [stage2_earlystop, stage2_reduce_lr, stage2_checkpoint, tb_callback2, per_class_cb2]

    print(f"\nStarting Stage 2 fine-tuning for up to {FINE_TUNE_EPOCHS} epochs...")
    history_stage2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks_stage2,
        class_weight=class_weights,
        verbose=1
    )

    # Save Stage 2 history plot
    plot_training_history(history_stage2, filename='training_history_stage2.png')

    # Final evaluation on test set
    print("\n" + "=" * 40)
    print("FINAL EVALUATION (after Stage 2)")
    print("=" * 40)

    eval_results = model.evaluate(test_dataset, verbose=0)
    metrics_names = model.metrics_names
    metrics_dict = dict(zip(metrics_names, eval_results))

    # Print metrics
    print(f"📊 Test Accuracy: {metrics_dict.get('accuracy', 0):.4f} ({metrics_dict.get('accuracy', 0)*100:.2f}%)")
    print(f"📊 Test Precision: {metrics_dict.get('precision', 0):.4f}")
    print(f"📊 Test Recall: {metrics_dict.get('recall', 0):.4f}")
    print(f"📊 Test AUC: {metrics_dict.get('auc', 0):.4f}")

    # Calculate F1 score
    precision = metrics_dict.get('precision', 0)
    recall = metrics_dict.get('recall', 0)
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    print(f"📊 F1-Score: {f1:.4f}")

    # Save final model (best of stage2)
    final_model_filename = 'models/kidney_stone_detection_model.h5'
    model.save(final_model_filename)

    # Save model metadata
    model_metadata = {
        "model_name": model.name,
        "accuracy": float(metrics_dict.get('accuracy', 0)),
        "precision": float(metrics_dict.get('precision', 0)),
        "recall": float(metrics_dict.get('recall', 0)),
        "f1_score": float(f1),
        "auc": float(metrics_dict.get('auc', 0)),
        "image_size": [224, 224],
        "class_names": class_names,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_trained_stage1": len(history_stage1.history.get('loss', [])),
        "epochs_trained_stage2": len(history_stage2.history.get('loss', [])),
        "batch_size": BATCH_SIZE,
        "transfer_learning": USE_TRANSFER_LEARNING
    }

    with open('models/model_metadata.json', 'w') as f:
        import json
        json.dump(model_metadata, f, indent=4)

    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📁 Model saved as: {final_model_filename}")
    print("📁 Best stage1 model saved as: models/best_kidney_stone_model_stage1.h5")
    print("📁 Best stage2 model saved as: models/best_kidney_stone_model_stage2.h5")
    print("📁 Model metadata saved as: models/model_metadata.json")
    print("📊 Training history saved as: training_history_stage1.png, training_history_stage2.png")
    print("=" * 60)

    # Combine histories (optional) - return the second stage history primarily
    return model, history_stage2, model_metadata


if __name__ == "__main__":
    # Run training
    model, history, metadata = train_model()