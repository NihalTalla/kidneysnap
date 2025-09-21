import os
import cv2
import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any, List

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import shared constants for consistency
try:
    from common.constants import (
        CLASS_NAMES,
        VALID_IMAGE_EXTENSIONS,
        MIN_IMAGE_SIZE,
        MIN_FILE_SIZE_BYTES,
        MAX_FILE_SIZE_BYTES,
        MIN_DATASET_SIZE_PER_CLASS,
        DEFAULT_SHUFFLE_BUFFER_SIZE,
        DEFAULT_PREFETCH_BUFFER_SIZE
    )
except Exception:
    # Fallback defaults if constants module is missing
    CLASS_NAMES = ['Normal', 'Kidney_Stone']
    VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    MIN_IMAGE_SIZE = (64, 64)
    MIN_FILE_SIZE_BYTES = 1024
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
    MIN_DATASET_SIZE_PER_CLASS = 500
    DEFAULT_SHUFFLE_BUFFER_SIZE = 1000
    DEFAULT_PREFETCH_BUFFER_SIZE = 1


class MedicalImagePreprocessor:
    """
    Simplified image preprocessor for binary classification (Normal vs Kidney_Stone).
    - Default target size is 224x224 (RGB).
    - Focuses on basic operations: load, resize, convert to RGB, normalize to [0,1],
      and optional channel-wise standardization.
    - Provides utilities for single-image inference preprocessing and creating
      TensorFlow datasets for training/validation.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        standardize: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            target_size: (width, height) to resize images to. Default (224, 224).
            normalize: If True, scale pixel values to [0, 1].
            standardize: If True, subtract mean and divide by std (channel-wise).
                         This is applied after normalization.
            mean: mean values for each channel (RGB) used if standardize=True.
            std: std values for each channel (RGB) used if standardize=True.
        """
        if not (isinstance(target_size, (tuple, list)) and len(target_size) == 2):
            raise ValueError("target_size must be a tuple/list of two integers (width, height).")
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.normalize = bool(normalize)
        self.standardize = bool(standardize)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def _load_image_bgr(self, image_path: str, as_gray: bool = False) -> np.ndarray:
        """
        Loads an image from disk using OpenCV (BGR ordering) and returns it as a NumPy array.

        Args:
            image_path: Path to image.
            as_gray: If True, load as grayscale and convert to 3-channel by replication.

        Returns:
            image array in BGR order (H, W, C) with dtype uint8.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if as_gray:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image (grayscale): {image_path}")
            # Convert single channel to 3-channel by stacking
            img = np.stack([img, img, img], axis=-1)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

        return img

    def preprocess_image(self, image_path: str, as_gray: bool = False) -> np.ndarray:
        """
        Preprocess a single image from disk for inference or ad-hoc use.

        Steps:
        - Load image (grayscale option supported).
        - Convert BGR -> RGB.
        - Resize to target_size (width, height).
        - Convert to float32 and normalize to [0,1] if requested.
        - Optionally standardize by channel using provided mean/std.

        Returns:
            np.ndarray of shape (height, width, 3), dtype float32.
        """
        img_bgr = self._load_image_bgr(image_path, as_gray=as_gray)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize expects (width, height) tuple in cv2.resize signature as (w,h)
        img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        img_float = img_resized.astype(np.float32)

        if self.normalize:
            img_float = img_float / 255.0

        if self.standardize:
            # Ensure mean/std are broadcastable over HxW
            img_float = (img_float - self.mean) / self.std

        return img_float

    def preprocess_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess an in-memory image array (H, W, C) or (H, W) grayscale.

        Returns:
            np.ndarray of shape (height, width, 3), dtype float32.
        """
        if image_array is None:
            raise ValueError("image_array is None")

        arr = np.asarray(image_array)

        if arr.ndim == 2:
            # grayscale -> convert to 3 channels
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA -> RGB
            arr = arr[:, :, :3]
        elif arr.ndim != 3 or arr.shape[2] not in (1, 3):
            raise ValueError("Unsupported image array shape. Expected (H,W), (H,W,1) or (H,W,3).")

        # If the input is in float [0,1], scale up to 0-255 before resizing to preserve precision
        if issubclass(arr.dtype.type, np.floating) and arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        # Convert to RGB if it's BGR-like (we cannot be certain, but assume input is RGB)
        # To be safe, we won't swap channels here; assume callers provide RGB arrays.
        resized = cv2.resize(arr, self.target_size, interpolation=cv2.INTER_AREA)
        img_float = resized.astype(np.float32)

        if self.normalize:
            img_float = img_float / 255.0

        if self.standardize:
            img_float = (img_float - self.mean) / self.std

        return img_float

    def preprocess_for_tflite(self, image_path: str, as_gray: bool = False, uint8: bool = False) -> np.ndarray:
        """
        Preprocess an image for TFLite inference.
        If uint8=True, returns uint8 array in [0,255], otherwise float32 (normalized or standardized).

        Returns:
            np.ndarray with shape (1, height, width, 3) suitable for model input.
        """
        img = self.preprocess_image(image_path, as_gray=as_gray)

        if uint8:
            # Convert back to uint8 [0,255]
            if self.normalize:
                img_out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img_out = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_out = img.astype(np.float32)

        # Add batch dimension
        return np.expand_dims(img_out, axis=0)

    # ---------------------------
    # New dataset validation tools
    # ---------------------------

    def _is_image_file(self, filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in set(ext.lower() for ext in VALID_IMAGE_EXTENSIONS)

    def _count_images_in_dir(self, data_dir: str) -> Dict[str, int]:
        """
        Count images per subdirectory (class). Returns dict class_name -> count.
        """
        counts = {}
        for entry in os.listdir(data_dir):
            path = os.path.join(data_dir, entry)
            if os.path.isdir(path):
                cnt = 0
                for root, _, files in os.walk(path):
                    for f in files:
                        if self._is_image_file(f):
                            cnt += 1
                counts[entry] = cnt
        return counts

    def validate_dataset(
        self,
        data_dir: str,
        min_size_per_class: int = MIN_DATASET_SIZE_PER_CLASS,
        check_integrity: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate a dataset directory before creating TF datasets.

        Checks:
            - Directory exists and is not empty
            - Contains expected class subdirectories (based on CLASS_NAMES)
            - Each class has at least min_size_per_class images (warning if not)
            - Basic image integrity checks (file size, able to open, dimensions)
            - Class balance warnings

        Returns:
            report: dict containing keys:
                - ok: bool
                - total_images: int
                - class_counts: dict
                - missing_classes: list
                - issues: list of strings/warnings
                - sample_problems: list of example problematic files (up to 20)
        """
        report: Dict[str, Any] = {
            "ok": True,
            "total_images": 0,
            "class_counts": {},
            "missing_classes": [],
            "issues": [],
            "sample_problems": []
        }

        if not os.path.isdir(data_dir):
            report["ok"] = False
            report["issues"].append(f"Data directory does not exist: {data_dir}")
            return report

        # Discover classes present
        present_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        present_lower = [p.lower() for p in present_classes]

        # Check expected classes (case-insensitive)
        missing = []
        for expected in CLASS_NAMES:
            if expected.lower() not in present_lower:
                missing.append(expected)
        if missing:
            report["missing_classes"] = missing
            report["issues"].append(f"Missing expected classes: {', '.join(missing)}")
            # Not fatal; allow validation to proceed for found classes
            report["ok"] = False

        # Count images per class
        class_counts = {}
        total = 0
        problematic_examples: List[str] = []

        for cls in present_classes:
            cls_path = os.path.join(data_dir, cls)
            cnt = 0
            for root, _, files in os.walk(cls_path):
                for fname in files:
                    if not self._is_image_file(fname):
                        continue
                    fpath = os.path.join(root, fname)
                    cnt += 1

                    # Basic file size checks
                    try:
                        fsize = os.path.getsize(fpath)
                    except Exception:
                        fsize = 0
                        problematic_examples.append(f"{fpath}: cannot read file size")
                        continue

                    if check_integrity:
                        if fsize < MIN_FILE_SIZE_BYTES:
                            problematic_examples.append(f"{fpath}: file too small ({fsize} bytes)")
                        elif fsize > MAX_FILE_SIZE_BYTES:
                            problematic_examples.append(f"{fpath}: file too large ({fsize} bytes)")
                        else:
                            # Try opening and checking dimensions
                            try:
                                img = cv2.imread(fpath)
                                if img is None:
                                    problematic_examples.append(f"{fpath}: unreadable or corrupted image (cv2.imread returned None)")
                                else:
                                    h, w = img.shape[:2]
                                    if w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
                                        problematic_examples.append(f"{fpath}: image too small ({w}x{h})")
                            except Exception as e:
                                problematic_examples.append(f"{fpath}: exception while opening ({e})")

            class_counts[cls] = cnt
            total += cnt

        report["total_images"] = total
        report["class_counts"] = class_counts

        # Check per-class minimum size thresholds
        for cls, cnt in class_counts.items():
            if cnt < min_size_per_class:
                report["issues"].append(f"Class '{cls}' has only {cnt} images (< recommended {min_size_per_class})")
                report["ok"] = False

        # Check class balance
        counts = list(class_counts.values())
        if counts:
            max_c = max(counts)
            min_c = min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 0
            if min_c == 0:
                report["issues"].append("One or more classes have zero images")
                report["ok"] = False
            elif max_c / float(min_c) > 3.0:
                report["issues"].append(f"Severe class imbalance detected (ratio {max_c}:{min_c})")
            elif max_c / float(min_c) > 2.0:
                report["issues"].append(f"Moderate class imbalance detected (ratio {max_c}:{min_c})")

        # Aggregate sample problems
        if problematic_examples:
            report["sample_problems"] = problematic_examples[:20]
            report["issues"].append(f"Found {len(problematic_examples)} sample integrity issues (showing up to 20)")

        # Final ok determination: if there are any issues flagged above, set ok False (unless only minor)
        if report["issues"]:
            # If the only issues are recorded missing_classes or small counts, keep ok as False already set.
            pass

        return report

    # ---------------------------
    # Dataset loading with validation and memory optimizations
    # ---------------------------

    def _determine_shuffle_buffer(self, data_dir: Optional[str], provided_batch: int) -> int:
        """
        Determine an appropriate shuffle buffer size based on dataset size and defaults.
        Uses DEFAULT_SHUFFLE_BUFFER_SIZE capped by total images.
        """
        buffer_default = DEFAULT_SHUFFLE_BUFFER_SIZE if DEFAULT_SHUFFLE_BUFFER_SIZE else 1000
        if data_dir and os.path.isdir(data_dir):
            counts = self._count_images_in_dir(data_dir)
            total = sum(counts.values()) if counts else 0
            if total <= 0:
                return buffer_default
            return int(min(buffer_default, max(100, total)))
        else:
            # fallback: scale with batch size
            return int(min(buffer_default, max(100, provided_batch * 10)))

    def _apply_performance_options(self, ds: tf.data.Dataset, cache_file: Optional[str], shuffle_buffer_size: Optional[int]) -> tf.data.Dataset:
        """
        Apply caching, shuffling and prefetching strategies in a safe manner:
            - If cache_file is provided, use disk cache to avoid large memory usage.
            - Otherwise use in-memory cache() (fast but may use memory).
            - Shuffle if shuffle_buffer_size provided.
            - Prefetch with AUTOTUNE.
        """
        AUTOTUNE = tf.data.AUTOTUNE
        if cache_file:
            try:
                ds = ds.cache(cache_file)
            except Exception:
                # fallback to in-memory cache
                ds = ds.cache()
        else:
            ds = ds.cache()

        if shuffle_buffer_size and shuffle_buffer_size > 0:
            try:
                ds = ds.shuffle(buffer_size=shuffle_buffer_size)
            except Exception:
                # If shuffle fails for some reason, ignore and continue
                pass

        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def load_dataset(
        self,
        data_dir: str,
        batch_size: int = 32,
        validation_split: float = 0.15,
        seed: int = 123,
        shuffle: bool = True,
        run_validation: bool = True,
    ):
        """
        Create TensorFlow datasets for training and validation using a directory structure:
            data_dir/
                Normal/
                Kidney_Stone/

        Uses tf.keras.utils.image_dataset_from_directory with label_mode='binary' for binary classification.

        Args:
            data_dir: Root directory containing class subfolders.
            batch_size: Batch size.
            validation_split: Fraction of data to reserve for validation (0 < validation_split < 1).
            seed: Random seed for shuffling/splitting.
            shuffle: Whether to shuffle the dataset.
            run_validation: If True, run dataset validation before creating datasets and print results.

        Returns:
            (train_dataset, val_dataset), both are tf.data.Dataset objects yielding (images, labels).
            Images are floats (if normalize=True) or original range depending on configuration.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        if not (0.0 < validation_split < 1.0):
            raise ValueError("validation_split must be between 0 and 1")

        # Optional dataset validation
        if run_validation:
            try:
                report = self.validate_dataset(data_dir)
                print("🔎 Dataset validation report:")
                for k, v in report.items():
                    if k in ("sample_problems", "class_counts"):
                        continue
                    print(f"  • {k}: {v}")
                if report.get("sample_problems"):
                    print("  • Sample problems (up to 5):")
                    for p in report["sample_problems"][:5]:
                        print(f"    - {p}")
            except Exception as e:
                print(f"⚠️  Dataset validation failed: {e}")

        # Determine shuffle buffer size based on dataset size
        shuffle_buffer_size = self._determine_shuffle_buffer(data_dir, batch_size) if shuffle else 0

        # Choose caching strategy: use disk cache if dataset is large to avoid memory blow-up
        cache_file = None
        try:
            counts = self._count_images_in_dir(data_dir)
            total_images = sum(counts.values()) if counts else 0
            # Heuristic: if > 5000 images, use disk-backed cache file
            if total_images > 5000:
                cache_file = os.path.join(data_dir, ".tf_data_cache")
        except Exception:
            cache_file = None

        # Create training dataset (subset="training")
        train_ds = image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='binary',
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=self.target_size,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        # Create validation dataset (subset="validation")
        val_ds = image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='binary',
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=self.target_size,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        # Define normalization layer (if requested)
        if self.normalize:
            normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
        else:
            # Identity lambda layer
            normalization_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))

        def map_fn(x, y):
            x = normalization_layer(x)
            if self.standardize:
                # Standardize using mean/std: mean/std in [0,1] space because normalization already applied
                mean = tf.constant(self.mean.reshape((1, 1, 1, 3)), dtype=tf.float32)
                std = tf.constant(self.std.reshape((1, 1, 1, 3)), dtype=tf.float32)
                x = (x - mean) / std
            return x, y

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(map_fn, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(map_fn, num_parallel_calls=AUTOTUNE)

        # Apply caching, shuffling, prefetching strategies
        train_ds = self._apply_performance_options(train_ds, cache_file if cache_file else None, shuffle_buffer_size if shuffle else 0)
        val_ds = self._apply_performance_options(val_ds, cache_file if cache_file else None, 0)

        return train_ds, val_ds

    def load_train_test_datasets(
        self,
        train_dir: str,
        test_dir: str,
        batch_size: int = 32,
        seed: int = 123,
        shuffle_train: bool = True,
        run_validation: bool = True,
    ):
        """
        Create TensorFlow datasets for training and testing using the project's expected directory structure:
            train_dir/
                Normal/
                Kidney_Stone/
            test_dir/
                Normal/
                Kidney_Stone/

        This does NOT use validation_split since training and testing are in separate directories.

        Args:
            train_dir: Directory containing training class subfolders.
            test_dir: Directory containing test class subfolders.
            batch_size: Batch size for both datasets.
            seed: Random seed for shuffling.
            shuffle_train: Whether to shuffle the training dataset. Test dataset is not shuffled.
            run_validation: If True, run dataset validation on train_dir and test_dir and print results.

        Returns:
            (train_dataset, test_dataset, class_names)
            - train_dataset, test_dataset: tf.data.Dataset objects yielding (images, labels)
            - class_names: list of class names inferred from the training directory
        """
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Optional validations
        if run_validation:
            try:
                print("🔎 Validating training directory...")
                train_report = self.validate_dataset(train_dir)
                print("  Training report - total_images:", train_report.get("total_images", 0))
                if train_report.get("issues"):
                    print("  Training issues:")
                    for issue in train_report["issues"]:
                        print(f"    - {issue}")
            except Exception as e:
                print(f"⚠️  Training dataset validation failed: {e}")

            try:
                print("🔎 Validating test directory...")
                test_report = self.validate_dataset(test_dir)
                print("  Test report - total_images:", test_report.get("total_images", 0))
                if test_report.get("issues"):
                    print("  Test issues:")
                    for issue in test_report["issues"]:
                        print(f"    - {issue}")
            except Exception as e:
                print(f"⚠️  Test dataset validation failed: {e}")

        # Determine shuffle buffer size for training
        shuffle_buffer_size = self._determine_shuffle_buffer(train_dir, batch_size) if shuffle_train else 0

        # Choose caching strategy
        cache_file_train = None
        cache_file_test = None
        try:
            train_counts = self._count_images_in_dir(train_dir)
            total_train = sum(train_counts.values()) if train_counts else 0
            if total_train > 5000:
                cache_file_train = os.path.join(train_dir, ".tf_data_cache")
            test_counts = self._count_images_in_dir(test_dir)
            total_test = sum(test_counts.values()) if test_counts else 0
            if total_test > 5000:
                cache_file_test = os.path.join(test_dir, ".tf_data_cache")
        except Exception:
            cache_file_train = cache_file_test = None

        # Use categorical labels to match typical model expectations (one-hot)
        train_ds = image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='categorical',
            seed=seed,
            image_size=self.target_size,
            batch_size=batch_size,
            shuffle=shuffle_train,
        )

        test_ds = image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='categorical',
            seed=seed,
            image_size=self.target_size,
            batch_size=batch_size,
            shuffle=False,
        )

        class_names = train_ds.class_names

        # Validate that class_names match expected canonical CLASS_NAMES (case-insensitive)
        expected_lower = [c.lower() for c in CLASS_NAMES]
        present_lower = [c.lower() for c in class_names]
        for expected in expected_lower:
            if expected not in present_lower:
                print(f"⚠️  Expected class '{expected}' not found in training classes: {class_names}")

        # Define normalization layer (if requested)
        if self.normalize:
            normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
        else:
            normalization_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))

        def map_fn(x, y):
            x = normalization_layer(x)
            if self.standardize:
                mean = tf.constant(self.mean.reshape((1, 1, 1, 3)), dtype=tf.float32)
                std = tf.constant(self.std.reshape((1, 1, 1, 3)), dtype=tf.float32)
                x = (x - mean) / std
            return x, y

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(map_fn, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(map_fn, num_parallel_calls=AUTOTUNE)

        # Apply cache/shuffle/prefetch strategies
        train_ds = self._apply_performance_options(train_ds, cache_file_train, shuffle_buffer_size if shuffle_train else 0)
        test_ds = self._apply_performance_options(test_ds, cache_file_test, 0)

        return train_ds, test_ds, class_names

    def get_dataset_statistics(self, dataset: Optional[tf.data.Dataset] = None, data_dir: Optional[str] = None, max_batches: int = 100) -> Dict[str, Any]:
        """
        Return detailed statistics about a dataset either from a tf.data.Dataset or by scanning a directory.

        Stats include:
            - total_images (if directory available)
            - class_distribution (if directory available)
            - sample_image_size_stats (avg/min/max height/width)
            - estimated_memory_bytes (float32 tensors for all images loaded into memory)
            - inspected_batches (number of batches iterated for tf.data.Dataset)
        """
        stats: Dict[str, Any] = {
            "total_images": None,
            "class_distribution": {},
            "avg_width": None,
            "avg_height": None,
            "min_width": None,
            "min_height": None,
            "max_width": None,
            "max_height": None,
            "inspected_batches": 0,
            "estimated_memory_bytes": None
        }

        widths: List[int] = []
        heights: List[int] = []
        total_images = 0

        # If a directory is provided, gather counts and basic file size stats
        if data_dir and os.path.isdir(data_dir):
            counts = self._count_images_in_dir(data_dir)
            stats["class_distribution"] = counts
            total_images = sum(counts.values())
            stats["total_images"] = total_images

            # Peek at up to 200 images to compute size stats
            sample_limit = 200
            sampled = 0
            for cls, cnt in counts.items():
                cls_path = os.path.join(data_dir, cls)
                for root, _, files in os.walk(cls_path):
                    for fname in files:
                        if sampled >= sample_limit:
                            break
                        if not self._is_image_file(fname):
                            continue
                        fpath = os.path.join(root, fname)
                        try:
                            img = cv2.imread(fpath)
                            if img is None:
                                continue
                            h, w = img.shape[:2]
                            widths.append(w)
                            heights.append(h)
                            sampled += 1
                        except Exception:
                            continue
                    if sampled >= sample_limit:
                        break
                if sampled >= sample_limit:
                    break

        # If a tf.data.Dataset is provided, inspect a limited number of batches
        if dataset is not None:
            inspected = 0
            imgs_seen = 0
            try:
                for batch in dataset.take(max_batches):
                    inspected += 1
                    x, y = batch
                    # x shape: (batch, height, width, channels)
                    if hasattr(x, "shape"):
                        b, h, w, c = x.shape
                        # account for possibly unknown dimensions
                        if h is None or w is None:
                            # try to infer from numpy
                            try:
                                arr = x.numpy()
                                h = arr.shape[1]
                                w = arr.shape[2]
                            except Exception:
                                continue
                        widths.extend([int(w)] * int(x.shape[0]))
                        heights.extend([int(h)] * int(x.shape[0]))
                        imgs_seen += int(x.shape[0])
                    else:
                        # Fallback: convert to numpy and inspect
                        try:
                            arr = x.numpy()
                            b, h, w, c = arr.shape
                            widths.extend([int(w)] * b)
                            heights.extend([int(h)] * b)
                            imgs_seen += b
                        except Exception:
                            continue
                stats["inspected_batches"] = inspected
                if stats["total_images"] is None:
                    stats["total_images"] = imgs_seen
                else:
                    # prefer directory-based total_images if both provided
                    pass
            except Exception:
                # If we cannot iterate the dataset (e.g., symbolically defined), skip
                pass

        # Compute width/height stats if available
        if widths and heights:
            widths_arr = np.array(widths)
            heights_arr = np.array(heights)
            stats["avg_width"] = float(np.mean(widths_arr))
            stats["avg_height"] = float(np.mean(heights_arr))
            stats["min_width"] = int(np.min(widths_arr))
            stats["min_height"] = int(np.min(heights_arr))
            stats["max_width"] = int(np.max(widths_arr))
            stats["max_height"] = int(np.max(heights_arr))

            # Estimate memory: assuming float32 after preprocessing (4 bytes per channel)
            est_total_images = stats["total_images"] if stats["total_images"] else len(widths)
            # Use average dims for estimate
            avg_w = stats["avg_width"]
            avg_h = stats["avg_height"]
            channels = 3
            if avg_w and avg_h and est_total_images:
                bytes_per_image = avg_w * avg_h * channels * 4.0  # float32
                estimated_memory = bytes_per_image * est_total_images
                stats["estimated_memory_bytes"] = int(estimated_memory)

        return stats

    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Returns the expected input shape for models: (height, width, channels).
        """
        # Keras/TensorFlow typically uses (height, width) ordering for image_size.
        height, width = self.target_size[1], self.target_size[0]
        return (height, width, 3)