#!/usr/bin/env python3
"""
Dataset Organization and Validation Tool for Kidney Stone Detection
Organizes, validates, and splits medical image datasets for training
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
import datetime
from tqdm import tqdm
import concurrent.futures

# Import shared constants for consistent naming and thresholds
from common.constants import (
    CLASS_NAMES,
    VALID_IMAGE_EXTENSIONS,
    MIN_IMAGE_SIZE,
    MIN_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_BYTES,
    DUPLICATE_SIMILARITY_THRESHOLD,
    REPORTS_DIR,
    DATE_FORMAT,
    MAX_WORKERS
)


def _average_hash(image_path: str, hash_size: int = 8) -> Optional[int]:
    """
    Compute a simple average hash for an image.
    Returns an integer representing the bits of the hash, or None on failure.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(img).flatten()
            avg = pixels.mean()
            bits = 1 * (pixels > avg)
            # Convert bits to integer
            bit_string = ''.join(str(int(b)) for b in bits)
            return int(bit_string, 2)
    except Exception:
        return None


def _hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two integer bitmasks."""
    return bin(a ^ b).count("1")


def _validate_image_worker(params: Tuple[str, Tuple[int, int], int, int]) -> Tuple[str, bool, str, Optional[Tuple[int, int]], Optional[int], Optional[int]]:
    """
    Worker function for validating a single image. Designed to be picklable for ProcessPoolExecutor.
    Returns tuple: (path, is_valid, reason, (width,height) or None, file_size_bytes or None, hash_int or None)
    """
    img_path, min_size, min_file_bytes, max_file_bytes = params
    try:
        # Basic file size checks
        try:
            file_size = os.path.getsize(img_path)
        except Exception:
            return (img_path, False, "cannot_read_file_size", None, None, None)

        if file_size < min_file_bytes:
            return (img_path, False, "file_too_small", None, file_size, None)
        if file_size > max_file_bytes:
            return (img_path, False, "file_too_large", None, file_size, None)

        # Attempt to open with PIL
        with Image.open(img_path) as img:
            mode = img.mode
            size = img.size
            if mode not in ['RGB', 'RGBA', 'L', 'P']:
                return (img_path, False, f"unsupported_mode:{mode}", None, file_size, None)
            if size[0] < min_size[0] or size[1] < min_size[1]:
                return (img_path, False, "too_small_dimensions", size, file_size, None)

        # Additional check with OpenCV (ensures not corrupted in some formats)
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            return (img_path, False, "opencv_cannot_read", size if 'size' in locals() else None, file_size, None)

        # Compute simple perceptual hash
        phash = _average_hash(img_path)

        return (img_path, True, "ok", size if 'size' in locals() else None, file_size, phash)
    except Exception as e:
        return (img_path, False, f"exception:{e}", None, None, None)


class DatasetOrganizer:
    """
    Comprehensive dataset organization tool for medical image datasets.
    Handles dataset validation, organization, splitting, and quality checks.
    """

    def __init__(self, base_path: str = "Dataset"):
        self.base_path = Path(base_path)
        # Use valid extensions from constants if available, otherwise fallback
        try:
            self.valid_extensions = set(ext.lower() for ext in VALID_IMAGE_EXTENSIONS)
        except Exception:
            self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # Use standardized class names from constants
        try:
            self.class_names = list(CLASS_NAMES)
        except Exception:
            self.class_names = ['Normal', 'Kidney_Stone']

        # Create base directories
        self.base_path.mkdir(exist_ok=True, parents=True)

    def scan_directory(self, directory: str) -> Dict[str, List[str]]:
        """
        Scan a directory and categorize images by class.

        Args:
            directory: Path to scan

        Returns:
            Dictionary mapping class names to lists of image paths
        """
        directory = Path(directory)
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return {}

        class_files = defaultdict(list)

        # Check if directory has class subdirectories
        subdirs = [d for d in directory.iterdir() if d.is_dir()]

        if subdirs:
            # Organized by subdirectories
            for subdir in subdirs:
                class_name = self._normalize_class_name(subdir.name)
                for file_path in subdir.iterdir():
                    if file_path.suffix.lower() in self.valid_extensions:
                        class_files[class_name].append(str(file_path))
        else:
            # All files in one directory - need manual classification
            for file_path in directory.iterdir():
                if file_path.suffix.lower() in self.valid_extensions:
                    # Try to infer class from filename
                    class_name = self._infer_class_from_filename(file_path.name)
                    class_files[class_name].append(str(file_path))

        return dict(class_files)

    def _normalize_class_name(self, name: str) -> str:
        """Normalize class names to standard format using canonical CLASS_NAMES."""
        name_lower = name.lower()

        # Stone-related keywords
        stone_keywords = ['kidney_stone', 'kidneystone', 'stone', 'stones', 'calculi', 'urolithiasis']
        if any(keyword in name_lower for keyword in stone_keywords):
            # Return canonical Kidney_Stone from CLASS_NAMES if present
            for cn in self.class_names:
                if cn.lower() == 'kidney_stone' or cn.replace(" ", "_").lower() == 'kidney_stone':
                    return cn
            return 'Kidney_Stone'

        # Normal-related keywords
        normal_keywords = ['normal', 'healthy', 'clean', 'clear']
        if any(keyword in name_lower for keyword in normal_keywords):
            for cn in self.class_names:
                if cn.lower() == 'normal':
                    return cn
            return 'Normal'

        # If the provided name matches one of the canonical names (case insensitive), return canonical
        for cn in self.class_names:
            if cn.lower() == name_lower.replace(" ", "_"):
                return cn

        # Default: Title-case the name to keep consistent folder naming
        return name.replace(" ", "_").title()

    def _infer_class_from_filename(self, filename: str) -> str:
        """Infer class from filename patterns."""
        filename_lower = filename.lower()

        # Stone indicators
        stone_indicators = ['stone', 'calculi', 'kidney_stone', 'kidneystone', 'abnormal', 'pathology']
        if any(indicator in filename_lower for indicator in stone_indicators):
            # Return canonical
            return self._normalize_class_name('Kidney_stone')

        # Normal indicators
        normal_indicators = ['normal', 'healthy', 'clean', 'clear']
        if any(indicator in filename_lower for indicator in normal_indicators):
            return self._normalize_class_name('Normal')

        # Default to normal if unclear
        return self._normalize_class_name('Normal')

    def validate_images(self,
                        image_paths: List[str],
                        min_size: Tuple[int, int] = MIN_IMAGE_SIZE,
                        use_multiprocessing: bool = True) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Validate image files and filter out corrupted or invalid images.

        Args:
            image_paths: List of image file paths
            min_size: Minimum image dimensions (width, height)
            use_multiprocessing: Whether to use multiprocessing for validation

        Returns:
            Tuple:
              - List of valid image paths
              - Dictionary mapping image_path -> info dict (size, file_size, hash, reason)
        """
        valid_images = []
        invalid_count = 0
        image_infos: Dict[str, Dict] = {}

        if not image_paths:
            return valid_images, image_infos

        params_list = [(p, min_size, MIN_FILE_SIZE_BYTES, MAX_FILE_SIZE_BYTES) for p in image_paths]

        # Choose executor: try ProcessPoolExecutor first for CPU bound hashing, else fallback to ThreadPoolExecutor
        executor_cls = concurrent.futures.ProcessPoolExecutor if use_multiprocessing else concurrent.futures.ThreadPoolExecutor

        # Limit workers
        max_workers = min(MAX_WORKERS if isinstance(MAX_WORKERS, int) else 4, len(params_list), 16)
        try:
            with executor_cls(max_workers=max_workers) as executor:
                # Map with tqdm progress bar
                futures = list(executor.map(_validate_image_worker, params_list))
                for result in tqdm(futures, desc="Validating images", unit="img"):
                    img_path, is_valid, reason, size, file_size, phash = result
                    if is_valid:
                        valid_images.append(img_path)
                        image_infos[img_path] = {
                            "size": size,
                            "file_size": file_size,
                            "hash": phash,
                            "reason": reason
                        }
                    else:
                        invalid_count += 1
                        image_infos[img_path] = {
                            "size": size,
                            "file_size": file_size,
                            "hash": phash,
                            "reason": reason
                        }
        except Exception:
            # Fallback to threaded executor if process-based fails (e.g., on some Windows contexts)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = list(executor.map(_validate_image_worker, params_list))
                for result in tqdm(futures, desc="Validating images", unit="img"):
                    img_path, is_valid, reason, size, file_size, phash = result
                    if is_valid:
                        valid_images.append(img_path)
                        image_infos[img_path] = {
                            "size": size,
                            "file_size": file_size,
                            "hash": phash,
                            "reason": reason
                        }
                    else:
                        invalid_count += 1
                        image_infos[img_path] = {
                            "size": size,
                            "file_size": file_size,
                            "hash": phash,
                            "reason": reason
                        }

        if invalid_count > 0:
            print(f"🔍 Filtered out {invalid_count} invalid images")

        return valid_images, image_infos

    def analyze_dataset_quality(self, class_files: Dict[str, List[str]]) -> Dict:
        """
        Analyze dataset quality and provide comprehensive statistics.

        Args:
            class_files: Dictionary mapping class names to image paths

        Returns:
            Dictionary containing dataset analysis
        """
        analysis = {
            'total_images': 0,
            'class_distribution': {},
            'image_sizes': [],  # list of (w,h)
            'file_sizes': [],  # bytes
            'file_formats': defaultdict(int),
            'quality_issues': [],
            'duplicates': []  # list of groups of duplicate file paths
        }

        # Collect all image infos for duplicate detection
        all_hashes = {}  # path -> hash
        size_widths = []
        size_heights = []

        for class_name, image_paths in class_files.items():
            valid_images, infos = self.validate_images(image_paths)
            analysis['class_distribution'][class_name] = len(valid_images)
            analysis['total_images'] += len(valid_images)

            # Gather properties for valid images
            for img_path in valid_images:
                info = infos.get(img_path, {})
                size = info.get("size")
                file_size = info.get("file_size")
                phash = info.get("hash")
                fmt = Path(img_path).suffix.lower()

                if size:
                    analysis['image_sizes'].append(size)
                    size_widths.append(size[0])
                    size_heights.append(size[1])
                if file_size:
                    analysis['file_sizes'].append(file_size)
                if fmt:
                    analysis['file_formats'][fmt] += 1
                if phash is not None:
                    all_hashes[img_path] = phash

        # Compute image dimension stats
        if analysis['image_sizes']:
            sizes = np.array(analysis['image_sizes'])
            analysis['avg_width'] = int(np.mean(sizes[:, 0]))
            analysis['avg_height'] = int(np.mean(sizes[:, 1]))
            analysis['min_width'] = int(np.min(sizes[:, 0]))
            analysis['min_height'] = int(np.min(sizes[:, 1]))
            analysis['max_width'] = int(np.max(sizes[:, 0]))
            analysis['max_height'] = int(np.max(sizes[:, 1]))
            analysis['width_variance'] = float(np.var(sizes[:, 0]))
            analysis['height_variance'] = float(np.var(sizes[:, 1]))

        # Compute file size distribution stats
        if analysis['file_sizes']:
            fs = np.array(analysis['file_sizes'])
            analysis['file_size_mean'] = float(np.mean(fs))
            analysis['file_size_median'] = float(np.median(fs))
            analysis['file_size_min'] = int(np.min(fs))
            analysis['file_size_max'] = int(np.max(fs))
            analysis['file_size_std'] = float(np.std(fs))

        # Duplicate detection based on perceptual hash similarity
        duplicates = []
        if all_hashes:
            paths = list(all_hashes.keys())
            hashes = [all_hashes[p] for p in paths]
            visited = set()
            n_bits = 8 * 8  # using 8x8 hash
            threshold = DUPLICATE_SIMILARITY_THRESHOLD if DUPLICATE_SIMILARITY_THRESHOLD else 0.95

            for i, p in enumerate(paths):
                if p in visited:
                    continue
                group = [p]
                hi = hashes[i]
                for j in range(i + 1, len(paths)):
                    q = paths[j]
                    if q in visited:
                        continue
                    hj = hashes[j]
                    if hi is None or hj is None:
                        continue
                    ham = _hamming_distance(hi, hj)
                    similarity = 1.0 - (ham / float(n_bits))
                    if similarity >= threshold:
                        group.append(q)
                        visited.add(q)
                if len(group) > 1:
                    duplicates.append({
                        "representative": group[0],
                        "group": group,
                        "group_size": len(group)
                    })
                    for gp in group:
                        visited.add(gp)

        analysis['duplicates'] = duplicates

        # Check for quality issues
        total_images = analysis['total_images']
        if total_images == 0:
            analysis['quality_issues'].append("No valid images found")
        else:
            # Check class imbalance
            class_counts = list(analysis['class_distribution'].values())
            if len(class_counts) > 1:
                max_count = max(class_counts)
                min_count = min(class_counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                if imbalance_ratio > 3:
                    analysis['quality_issues'].append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f}:1)")
                elif imbalance_ratio > 2:
                    analysis['quality_issues'].append(f"Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)")

            # Check dataset size relative thresholds
            if total_images < 100:
                analysis['quality_issues'].append("Very small dataset (<100 images)")
            elif total_images < 500:
                analysis['quality_issues'].append("Small dataset (<500 images)")

            # Duplicate warning
            if duplicates:
                analysis['quality_issues'].append(f"Found {len(duplicates)} potential duplicate groups")

        return analysis

    def create_reports_directory(self) -> Path:
        """
        Create a timestamped reports directory under the dataset base path.
        Format: Dataset/reports/YYYYMMDD-HHMM/
        Returns the Path to the created reports directory.
        """
        timestamp = datetime.datetime.now().strftime(DATE_FORMAT if DATE_FORMAT else "%Y%m%d-%H%M")
        reports_root = self.base_path / (REPORTS_DIR if REPORTS_DIR else "reports")
        reports_root.mkdir(parents=True, exist_ok=True)
        report_dir = reports_root / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def generate_quality_report(self, analysis: Dict, report_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Generate JSON and Markdown quality reports based on analysis dictionary.

        Args:
            analysis: Analysis dict returned by analyze_dataset_quality
            report_dir: Optional directory to save the report. If None, creates a timestamped reports dir.

        Returns:
            Tuple of (json_path, markdown_path)
        """
        if report_dir:
            report_path = Path(report_dir)
            report_path.mkdir(parents=True, exist_ok=True)
        else:
            report_path = self.create_reports_directory()

        # JSON report
        json_path = report_path / "quality_report.json"
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(analysis, jf, indent=2, default=str)

        # Markdown report
        md_lines = []
        md_lines.append(f"# Dataset Quality Report")
        md_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md_lines.append("## Summary")
        md_lines.append(f"- Total images: {analysis.get('total_images', 0)}")
        md_lines.append("\n## Class Distribution\n")
        md_lines.append("| Class | Count |")
        md_lines.append("|---|---:|")
        for cn, cnt in analysis.get('class_distribution', {}).items():
            md_lines.append(f"| {cn} | {cnt} |")

        md_lines.append("\n## Image Size Statistics\n")
        if 'avg_width' in analysis:
            md_lines.append(f"- Average size: {analysis.get('avg_width')} x {analysis.get('avg_height')}")
            md_lines.append(f"- Width variance: {analysis.get('width_variance', 0):.2f}")
            md_lines.append(f"- Height variance: {analysis.get('height_variance', 0):.2f}")

        md_lines.append("\n## File Size Statistics (bytes)\n")
        if 'file_size_mean' in analysis:
            md_lines.append(f"- Mean: {analysis.get('file_size_mean'):.1f}")
            md_lines.append(f"- Median: {analysis.get('file_size_median'):.1f}")
            md_lines.append(f"- Min: {analysis.get('file_size_min')}")
            md_lines.append(f"- Max: {analysis.get('file_size_max')}")
            md_lines.append(f"- Std: {analysis.get('file_size_std'):.1f}")

        md_lines.append("\n## File Formats\n")
        md_lines.append("| Format | Count |")
        md_lines.append("|---|---:|")
        for fmt, cnt in analysis.get('file_formats', {}).items():
            md_lines.append(f"| {fmt} | {cnt} |")

        md_lines.append("\n## Quality Issues\n")
        if analysis.get('quality_issues'):
            for issue in analysis.get('quality_issues', []):
                md_lines.append(f"- {issue}")
        else:
            md_lines.append("- No major quality issues detected")

        md_lines.append("\n## Duplicate Groups\n")
        duplicates = analysis.get('duplicates', [])
        if duplicates:
            md_lines.append(f"Found {len(duplicates)} potential duplicate groups. Example groups:")
            for group in duplicates[:10]:
                md_lines.append(f"- Group size: {group.get('group_size')}, Representative: {group.get('representative')}")
                for member in group.get('group', [])[:5]:
                    md_lines.append(f"  - {member}")
        else:
            md_lines.append("- No duplicate groups detected")

        md_path = report_path / "quality_report.md"
        with open(md_path, 'w', encoding='utf-8') as mf:
            mf.write("\n".join(md_lines))

        print(f"📄 Quality reports saved to: {report_path}")
        return json_path, md_path

    def organize_dataset(self,
                         source_dir: str,
                         train_split: float = 0.7,
                         val_split: float = 0.15,
                         test_split: float = 0.15,
                         copy_files: bool = True,
                         seed: int = 42) -> bool:
        """
        Organize dataset into train/val/test splits with proper directory structure.

        Args:
            source_dir: Source directory containing images
            train_split: Fraction for training set
            val_split: Fraction for validation set  
            test_split: Fraction for test set
            copy_files: If True, copy files; if False, move files
            seed: Random seed for reproducible splits

        Returns:
            True if successful, False otherwise
        """
        print(f"📁 Organizing dataset from: {source_dir}")

        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            print("❌ Train/Val/Test splits must sum to 1.0")
            return False

        # Scan source directory
        class_files = self.scan_directory(source_dir)
        if not class_files:
            print("❌ No valid images found in source directory")
            return False

        # Analyze dataset
        analysis = self.analyze_dataset_quality(class_files)
        self._print_dataset_analysis(analysis)

        # Optionally generate quality report
        try:
            self.generate_quality_report(analysis)
        except Exception as e:
            print(f"⚠️  Could not generate quality report: {e}")

        # Set random seed for reproducible splits
        random.seed(seed)

        # Create target directories
        splits = ['Train', 'Val', 'Test']
        for split in splits:
            split_dir = self.base_path / split
            split_dir.mkdir(exist_ok=True, parents=True)
            for class_name in self.class_names:
                (split_dir / class_name).mkdir(exist_ok=True, parents=True)

        # Organize files for each class
        total_moved = 0

        # Iterate classes with a progress bar for better UX on large datasets
        for class_name, image_paths in tqdm(class_files.items(), desc="Classes", unit="class"):
            # Validate images first (using multiprocessing for speed)
            valid_images, _ = self.validate_images(image_paths, use_multiprocessing=True)

            if not valid_images:
                print(f"⚠️  No valid images found for class: {class_name}")
                continue

            # Normalize class name (ensure canonical)
            normalized_class = self._normalize_class_name(class_name)

            # Shuffle images
            random.shuffle(valid_images)

            # Calculate split indices
            n_images = len(valid_images)
            n_train = int(n_images * train_split)
            n_val = int(n_images * val_split)

            # Split images
            train_images = valid_images[:n_train]
            val_images = valid_images[n_train:n_train + n_val]
            test_images = valid_images[n_train + n_val:]

            # Move/copy files with progress bar
            splits_data = [
                ('Train', train_images),
                ('Val', val_images),
                ('Test', test_images)
            ]

            for split_name, images in splits_data:
                target_dir = self.base_path / split_name / normalized_class
                # Use tqdm to show progress of file operations per class/split
                for img_path in tqdm(images, desc=f"{normalized_class}::{split_name}", leave=False, unit="img"):
                    source_path = Path(img_path)
                    target_path = target_dir / source_path.name

                    # Handle name conflicts
                    counter = 1
                    while target_path.exists():
                        stem = source_path.stem
                        suffix = source_path.suffix
                        target_path = target_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                    try:
                        if copy_files:
                            shutil.copy2(source_path, target_path)
                        else:
                            shutil.move(str(source_path), target_path)
                        total_moved += 1
                    except Exception as e:
                        print(f"⚠️  Error processing {source_path}: {e}")

            print(f"✅ Organized {normalized_class}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

        # Save organization metadata
        metadata = {
            'organization_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_directory': str(source_dir),
            'total_images_processed': total_moved,
            'splits': {
                'train': train_split,
                'val': val_split,
                'test': test_split
            },
            'seed': seed,
            'copy_files': copy_files,
            'dataset_analysis': analysis
        }

        with open(self.base_path / 'organization_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n✅ Dataset organization completed!")
        print(f"📊 Total images processed: {total_moved}")
        print(f"📁 Organized dataset saved to: {self.base_path}")
        print(f"📄 Metadata saved to: {self.base_path / 'organization_metadata.json'}")

        return True

    def _print_dataset_analysis(self, analysis: Dict):
        """Print dataset analysis in a formatted way."""
        print("\n" + "=" * 50)
        print("📊 DATASET ANALYSIS")
        print("=" * 50)

        print(f"Total Images: {analysis.get('total_images', 0)}")

        print("\nClass Distribution:")
        for class_name, count in analysis.get('class_distribution', {}).items():
            percentage = (count / analysis['total_images'] * 100) if analysis.get('total_images', 0) > 0 else 0
            print(f"  • {class_name}: {count} images ({percentage:.1f}%)")

        if 'avg_width' in analysis:
            print(f"\nImage Dimensions:")
            print(f"  • Average: {analysis.get('avg_width')}x{analysis.get('avg_height')}")
            print(f"  • Range: {analysis.get('min_width')}x{analysis.get('min_height')} to {analysis.get('max_width')}x{analysis.get('max_height')}")
            print(f"  • Width Variance: {analysis.get('width_variance'):.2f}")
            print(f"  • Height Variance: {analysis.get('height_variance'):.2f}")

        if analysis.get('file_formats'):
            print(f"\nFile Formats:")
            for fmt, count in analysis.get('file_formats', {}).items():
                print(f"  • {fmt}: {count} files")

        if 'file_size_mean' in analysis:
            print(f"\nFile Size (bytes):")
            print(f"  • Mean: {analysis.get('file_size_mean'):.1f}")
            print(f"  • Median: {analysis.get('file_size_median'):.1f}")
            print(f"  • Min: {analysis.get('file_size_min')}")
            print(f"  • Max: {analysis.get('file_size_max')}")
            print(f"  • Std: {analysis.get('file_size_std'):.1f}")

        if analysis.get('duplicates'):
            print(f"\n🔁 Potential duplicate groups: {len(analysis.get('duplicates'))}")
            for dup in analysis.get('duplicates', [])[:5]:
                print(f"  • Group size {dup.get('group_size')}: Representative: {dup.get('representative')}")

        if analysis.get('quality_issues'):
            print(f"\n⚠️  Quality Issues:")
            for issue in analysis.get('quality_issues', []):
                print(f"  • {issue}")
        else:
            print(f"\n✅ No major quality issues detected")

        print("=" * 50)

    def create_sample_dataset(self, n_samples_per_class: int = 50):
        """
        Create a sample dataset for testing purposes.

        Args:
            n_samples_per_class: Number of sample images per class
        """
        print(f"🎨 Creating sample dataset with {n_samples_per_class} images per class...")

        # Create directories
        sample_dir = Path("Sample_Dataset")
        for class_name in self.class_names:
            (sample_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Generate sample images
        np.random.seed(42)

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = sample_dir / class_name

            for i in range(n_samples_per_class):
                # Create different patterns for each class
                if 'stone' in class_name.lower():
                    # Stone-like patterns: more contrast, irregular shapes
                    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                    # Add some bright spots to simulate stones
                    for _ in range(5):
                        x, y = np.random.randint(50, 174, 2)
                        size = np.random.randint(10, 30)
                        cv2.circle(img, (x, y), size, (255, 255, 255), -1)
                else:
                    # Normal kidney patterns: smoother, more uniform
                    img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                    # Add smooth gradients
                    overlay = np.random.normal(128, 30, (224, 224, 3))
                    img = np.clip(img + overlay, 0, 255).astype(np.uint8)

                # Save image
                filename = f"{class_name.lower()}_{i+1:03d}.png"
                cv2.imwrite(str(class_dir / filename), img)

        print(f"✅ Sample dataset created in: {sample_dir}")
        print(f"📊 Generated {n_samples_per_class * len(self.class_names)} total images")

    def verify_dataset_structure(self) -> bool:
        """
        Verify that the organized dataset has the correct structure.

        Returns:
            True if structure is valid, False otherwise
        """
        print("🔍 Verifying dataset structure...")

        required_splits = ['Train', 'Test']
        optional_splits = ['Val']
        issues = []

        for split in required_splits:
            split_dir = self.base_path / split
            if not split_dir.exists():
                issues.append(f"Missing required split: {split}")
                continue

            for class_name in self.class_names:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    issues.append(f"Missing class directory: {split}/{class_name}")
                else:
                    # Count images
                    images = [f for f in class_dir.iterdir()
                              if f.suffix.lower() in self.valid_extensions]
                    if len(images) == 0:
                        issues.append(f"No images in: {split}/{class_name}")
                    else:
                        print(f"✅ {split}/{class_name}: {len(images)} images")

        if issues:
            print("\n❌ Dataset structure issues:")
            for issue in issues:
                print(f"  • {issue}")
            return False
        else:
            print("\n✅ Dataset structure is valid!")
            return True


def main():
    """Main function demonstrating dataset organization usage."""
    print("🏥 Kidney Stone Detection - Dataset Organizer")
    print("=" * 60)

    organizer = DatasetOrganizer()

    # Check if we have existing data to organize
    potential_sources = [
        "raw_data",
        "original_images",
        "unorganized_data",
        "kidney_images"
    ]

    source_found = None
    for source in potential_sources:
        if os.path.exists(source):
            source_found = source
            break

    if source_found:
        print(f"📁 Found potential source directory: {source_found}")
        response = input("Would you like to organize this dataset? (y/n): ").lower().strip()

        if response == 'y':
            success = organizer.organize_dataset(
                source_dir=source_found,
                train_split=0.7,
                val_split=0.15,
                test_split=0.15,
                copy_files=True  # Keep originals safe
            )

            if success:
                organizer.verify_dataset_structure()
        else:
            print("📝 Skipping dataset organization")
    else:
        print("📝 No source directories found in common locations")
        print("💡 You can organize a dataset by calling:")
        print("   organizer.organize_dataset('your_source_directory')")

        # Ask if user wants to create sample data
        response = input("\nWould you like to create a sample dataset for testing? (y/n): ").lower().strip()
        if response == 'y':
            organizer.create_sample_dataset(n_samples_per_class=30)

            # Organize the sample dataset
            print("\n📁 Organizing sample dataset...")
            organizer.organize_dataset(
                source_dir="Sample_Dataset",
                train_split=0.7,
                val_split=0.15,
                test_split=0.15,
                copy_files=True
            )

            organizer.verify_dataset_structure()

    print("\n✅ Dataset organizer completed!")


if __name__ == "__main__":
    main()