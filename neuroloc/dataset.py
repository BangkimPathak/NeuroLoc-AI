"""
Module 1: Data Ingestion, Deduplication & Preprocessing
Handles Kaggle dataset download, dHash perceptual deduplication across splits,
Ben Graham skull contour cropping, CLAHE contrast enhancement,
and high-throughput tf.data input pipelines.
"""

import os
import glob
import logging
from typing import List, Tuple, Dict, Optional, Generator, Union
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
import imagehash  # type: ignore[import]
import tensorflow as tf

logger = logging.getLogger(__name__)

# Canonical Class Definitions (Masoud Nickparvar Dataset)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


def download_kaggle_dataset(
        
    dataset_slug: str = "masoudnickparvar/brain-tumor-mri-dataset",
    destination_dir: str = ".",
    force: bool = False,
) -> Tuple[str, str]:
    """
    Download and extract Kaggle Brain Tumor MRI dataset via Kaggle API.
    If 'Training' and 'Testing' directories already exist, downloads are skipped.

    Args:
        dataset_slug: Kaggle dataset identifier.
        destination_dir: Target download and extraction root.
        force: Whether to re-download even if directories exist.

    Returns:
        Tuple of (train_dir, test_dir).
    """
    train_dir = os.path.join(destination_dir, "Training")
    test_dir = os.path.join(destination_dir, "Testing")

    if not force and os.path.exists(train_dir) and os.path.exists(test_dir):
        logger.info("Dataset found locally at: %s and %s", train_dir, test_dir)
        return train_dir, test_dir

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import]
        api = KaggleApi()
        api.authenticate()
        logger.info("Downloading Kaggle dataset '%s' to '%s'...", dataset_slug, destination_dir)
        api.dataset_download_files(dataset_slug, path=destination_dir, unzip=True)
        logger.info("Download and decompression complete.")
    except Exception as exc:
        logger.warning(
            "Automatic Kaggle download failed (%s). "
            "Please ensure ~/.kaggle/kaggle.json exists or download manually.", exc
        )

    return train_dir, test_dir


# =========================================================================
# Perceptual Hashing & Split Deduplication (imagehash.dhash)
# =========================================================================

def compute_image_dhash(image_path_or_array: Union[str, np.ndarray], hash_size: int = 8) -> imagehash.ImageHash:
    """
    Compute 64-bit difference hash (dHash) for an image to detect duplicates and leakages.

    Args:
        image_path_or_array: File path string or uint8 numpy array.
        hash_size: Number of hash frequency bins (default 8 yields 64-bit hash).

    Returns:
        imagehash.ImageHash instance.
    """
    if isinstance(image_path_or_array, str):
        with Image.open(image_path_or_array) as pil_img:
            return imagehash.dhash(pil_img, hash_size=hash_size)
    elif isinstance(image_path_or_array, np.ndarray):
        pil_img = Image.fromarray(image_path_or_array)
        return imagehash.dhash(pil_img, hash_size=hash_size)
    else:
        raise TypeError(f"Unsupported image type: {type(image_path_or_array)}")


def audit_and_deduplicate_dataset(
    train_dir: str,
    test_dir: str,
    hamming_threshold: int = 2,
    quarantine_duplicates: bool = False,
    quarantine_dir: Optional[str] = None,
) -> Dict[str, any]:
    """
    Audit training and testing splits for cross-split data leakage and internal duplicates
    using difference hashing (dHash).

    Args:
        train_dir: Directory containing Training subfolders.
        test_dir: Directory containing Testing subfolders.
        hamming_threshold: Maximum Hamming distance to consider two images duplicates (<=2 is standard).
        quarantine_duplicates: If True, moves leaking test images to quarantine_dir.
        quarantine_dir: Folder to move leaking images into.

    Returns:
        Audit report containing summary counts, leaking paths, and duplicate clusters.
    """
    logger.info("Computing dHash for all training images in %s...", train_dir)
    train_hashes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    train_count = 0

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(cls_dir, img_name)
                try:
                    h = str(compute_image_dhash(fpath))
                    train_hashes[h].append((fpath, cls))
                    train_count += 1
                except Exception as e:
                    logger.debug("Failed hashing %s: %s", fpath, e)

    logger.info("Hashed %d training images across %d unique hashes.", train_count, len(train_hashes))

    logger.info("Auditing testing images in %s for cross-split leakage...", test_dir)
    test_count = 0
    exact_leakage_count = 0
    near_leakage_count = 0
    leakage_records = []

    # Pre-parse train hash objects for fast hamming comparison if threshold > 0
    train_hash_objs = [(imagehash.hex_to_hash(h_str), entries) for h_str, entries in train_hashes.items()]

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                test_path = os.path.join(cls_dir, img_name)
                try:
                    test_hash = compute_image_dhash(test_path)
                    test_count += 1

                    # Check exact hash match
                    test_hash_str = str(test_hash)
                    if test_hash_str in train_hashes:
                        exact_leakage_count += 1
                        matching_train = train_hashes[test_hash_str]
                        leakage_records.append({
                            "test_image": test_path,
                            "test_class": cls,
                            "matching_train": matching_train,
                            "hamming_distance": 0,
                            "type": "exact_leakage",
                        })
                        continue

                    # Check near duplicates within Hamming distance
                    if hamming_threshold > 0:
                        for tr_h, entries in train_hash_objs:
                            dist = test_hash - tr_h
                            if dist <= hamming_threshold:
                                near_leakage_count += 1
                                leakage_records.append({
                                    "test_image": test_path,
                                    "test_class": cls,
                                    "matching_train": entries,
                                    "hamming_distance": dist,
                                    "type": "near_leakage",
                                })
                                break
                except Exception as e:
                    logger.debug("Failed hashing test %s: %s", test_path, e)

    total_leakage = exact_leakage_count + near_leakage_count
    logger.info(
        "Audit Complete: Evaluated %d test images. Found %d exact leaks and %d near-duplicate leaks.",
        test_count, exact_leakage_count, near_leakage_count
    )

    if quarantine_duplicates and leakage_records:
        if quarantine_dir is None:
            quarantine_dir = os.path.join(os.path.dirname(test_dir), "Quarantine_Leakage")
        os.makedirs(quarantine_dir, exist_ok=True)
        import shutil
        for rec in leakage_records:
            dst = os.path.join(quarantine_dir, os.path.basename(rec["test_image"]))
            try:
                shutil.move(rec["test_image"], dst)
            except Exception as e:
                logger.warning("Failed moving %s to quarantine: %s", rec["test_image"], e)
        logger.info("Moved %d leaking images to '%s'", len(leakage_records), quarantine_dir)

    return {
        "train_images_analyzed": train_count,
        "test_images_analyzed": test_count,
        "exact_leakage_count": exact_leakage_count,
        "near_leakage_count": near_leakage_count,
        "total_leakage_count": total_leakage,
        "leakage_rate": total_leakage / max(test_count, 1),
        "leakage_records": leakage_records,
    }


# =========================================================================
# Image Preprocessing: Ben Graham Skull-Crop, CLAHE, & Cranial Masking
# =========================================================================

def extract_cranial_brain_mask(
    image: np.ndarray,
    tolerance: int = 15,
    min_area_fraction: float = 0.15,
) -> np.ndarray:
    """
    Extract binary mask of the cranial vault / brain parenchyma (255 inside head, 0 outside head).
    Guarantees all downstream XAI heatmaps, bounding boxes, and centroids are strictly
    contained within the anatomical skull boundaries.

    Args:
        image: BGR, RGB, or Grayscale image array.
        tolerance: Intensity cutoff for tissue/bone thresholding.
        min_area_fraction: Minimum cranial contour fraction.

    Returns:
        uint8 binary mask of shape (H, W) where 255 indicates intracranial tissue.
    """
    if image is None or image.size == 0:
        return np.zeros((224, 224), dtype=np.uint8)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, tolerance, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)

    if not contours:
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.42), int(h * 0.45)), 0, 0, 360, 255, -1)
        return mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area_fraction * (h * w):
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.42), int(h * 0.45)), 0, 0, 360, 255, -1)
        return mask

    cv2.drawContours(mask, [c], -1, 255, -1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    return mask


def ben_graham_skull_crop(
    image: np.ndarray,
    tolerance: int = 15,
    min_area_fraction: float = 0.20,
) -> np.ndarray:
    """
    Ben Graham circular / extreme contour skull-crop for MRI scans.
    Removes black border scanner margins, markers, and non-parenchymal air.
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, tolerance, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area = image.shape[0] * image.shape[1]

    if area < min_area_fraction * total_area:
        return image

    x, y, w, h = cv2.boundingRect(c)
    pad_x = int(0.02 * w)
    pad_y = int(0.02 * h)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    cropped = image[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else image


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances contrast across soft brain tissues without amplifying scanner noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        return clahe.apply(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        enhanced_lab = cv2.merge(lab_planes)
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        return image


def preprocess_mri_image(
    image_path_or_array: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    apply_crop: bool = True,
    apply_contrast: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """
    Complete clinical preprocessing pipeline:
    1. Read / parse image (RGB format)
    2. Ben Graham skull contour crop
    3. CLAHE adaptive contrast normalization
    4. Resize with bicubic interpolation
    5. Returns float32 in range [0, 255] (or [0, 1] if normalize=True).

    Args:
        image_path_or_array: Filepath or numpy image.
        target_size: Target (H, W) for network input.
        apply_crop: Whether to perform Ben Graham skull crop.
        apply_contrast: Whether to apply CLAHE contrast enhancement.
        normalize: Whether to scale pixel values to [0.0, 1.0]. Defaults to False
                   to match EfficientNet's internal Rescaling(1/255) layer.

    Returns:
        Preprocessed float32 array of shape (H, W, 3).
    """
    if isinstance(image_path_or_array, str):
        bgr = cv2.imread(image_path_or_array)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image from: {image_path_or_array}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = image_path_or_array.copy()
        if len(rgb.shape) == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

    if apply_crop:
        rgb = ben_graham_skull_crop(rgb)

    if apply_contrast:
        rgb = apply_clahe(rgb)

    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_CUBIC)

    if normalize:
        return (resized.astype(np.float32) / 255.0)
    return resized.astype(np.float32)


# =========================================================================
# High-Throughput tf.data Input Pipeline
# =========================================================================

def get_image_paths_and_labels(data_dir: str) -> Tuple[List[str], List[int]]:
    """Scan data_dir for class subfolders and collect all image paths and labels."""
    filepaths = []
    labels = []

    for cls_name, cls_idx in LABEL_MAP.items():
        cls_folder = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_folder):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for fpath in glob.glob(os.path.join(cls_folder, ext)):
                filepaths.append(fpath)
                labels.append(cls_idx)

    return filepaths, labels


def _py_preprocess_image(fpath_bytes) -> np.ndarray:
    """Wrapper for cv2 / numpy preprocessing inside tf.py_function."""
    if hasattr(fpath_bytes, "numpy"):
        fpath_raw = fpath_bytes.numpy()
    else:
        fpath_raw = fpath_bytes

    if isinstance(fpath_raw, bytes):
        fpath = fpath_raw.decode("utf-8")
    else:
        fpath = str(fpath_raw)
    return preprocess_mri_image(fpath, target_size=(224, 224), apply_crop=True, apply_contrast=True, normalize=False)


def build_augmentation_model() -> tf.keras.Sequential:
    """TensorFlow native augmentation pipeline for robust training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),  # +/- ~15 degrees
        tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08)),
        tf.keras.layers.RandomContrast(0.15),
    ], name="data_augmentation")


def create_dataset(
    data_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    shuffle_buffer: int = 1000,
    one_hot: bool = False,
    cache: bool = True,
    sample_limit: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Construct high-throughput tf.data.Dataset pipeline with AUTOTUNE prefetching.

    Args:
        data_dir: Directory containing class subfolders.
        batch_size: Batch size.
        image_size: Target (H, W).
        is_training: Whether to shuffle and apply data augmentations.
        shuffle_buffer: Buffer size for random shuffling.
        one_hot: If True, return one-hot encoded labels of shape (num_classes,).
        cache: Whether to cache elements in memory.
        sample_limit: Optional integer to limit dataset size (useful for quick verification).

    Returns:
        tf.data.Dataset yielding (images, labels) batches.
    """
    filepaths, labels = get_image_paths_and_labels(data_dir)
    if not filepaths:
        raise ValueError(f"No valid images found in directory: {data_dir}")

    if sample_limit is not None and sample_limit < len(filepaths):
        indices = np.random.RandomState(42).permutation(len(filepaths))[:sample_limit]
        filepaths = [filepaths[i] for i in indices]
        labels = [labels[i] for i in indices]

    logger.info("Found %d images across %d classes in '%s'.", len(filepaths), NUM_CLASSES, data_dir)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(filepaths), shuffle_buffer), seed=42)

    def _load_and_preprocess(path_tensor, label_tensor):
        img = tf.py_function(
            func=_py_preprocess_image,
            inp=[path_tensor],
            Tout=tf.float32,
        )
        img.set_shape([image_size[0], image_size[1], 3])
        if one_hot:
            label_out = tf.one_hot(label_tensor, depth=NUM_CLASSES)
        else:
            label_out = label_tensor
        return img, label_out

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if is_training:
        aug = build_augmentation_model()
        ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


# =========================================================================
# Extensible Support: BraTS (NIfTI) & TCIA (DICOM) Hooks
# =========================================================================

def extract_2d_slices_from_nifti(
    nii_path: str,
    plane: str = "axial",
    min_intensity: float = 0.0,
    stride: int = 1,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Hook to extract 2D axial/coronal/sagittal slices from 3D BraTS NIfTI volumes (.nii / .nii.gz).

    Args:
        nii_path: Path to NIfTI file.
        plane: 'axial', 'coronal', or 'sagittal'.
        min_intensity: Minimum mean slice intensity threshold to skip non-brain slices.
        stride: Slice sampling frequency.

    Yields:
        Tuple of (slice_index, 2D float32 normalized slice).
    """
    try:
        import nibabel as nib  # type: ignore[import]
    except ImportError:
        logger.error("nibabel is required for NIfTI volume processing. Install via: pip install nibabel")
        return

    nii_img = nib.load(nii_path)
    volume = nii_img.get_fdata(dtype=np.float32)

    # Determine slice axis
    axis_map = {"sagittal": 0, "coronal": 1, "axial": 2}
    axis = axis_map.get(plane.lower(), 2)

    total_slices = volume.shape[axis]
    for idx in range(0, total_slices, stride):
        if axis == 0:
            sl = volume[idx, :, :]
        elif axis == 1:
            sl = volume[:, idx, :]
        else:
            sl = volume[:, :, idx]

        if np.mean(sl) <= min_intensity:
            continue

        # Robust 1st - 99th percentile windowing
        p1, p99 = np.percentile(sl, (1.0, 99.0))
        if p99 > p1:
            sl = np.clip((sl - p1) / (p99 - p1), 0.0, 1.0)
        else:
            sl = np.zeros_like(sl)

        yield idx, sl


def extract_2d_slices_from_dicom(
    dicom_dir: str,
    window_center: int = 40,
    window_width: int = 80,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Hook to load and window 2D slices from TCIA DICOM series.

    Args:
        dicom_dir: Directory containing .dcm files.
        window_center: Hounsfield Unit (HU) window center (e.g. 40 for brain).
        window_width: HU window width (e.g. 80 for brain).

    Yields:
        Tuple of (filename, normalized float32 2D image).
    """
    try:
        import pydicom  # type: ignore[import]
    except ImportError:
        logger.error("pydicom is required for DICOM processing. Install via: pip install pydicom")
        return

    for fname in sorted(os.listdir(dicom_dir)):
        if fname.lower().endswith(".dcm"):
            fpath = os.path.join(dicom_dir, fname)
            try:
                ds = pydicom.dcmread(fpath)
                pixels = ds.pixel_array.astype(np.float32)
                # Rescale slope and intercept if available
                slope = getattr(ds, "RescaleSlope", 1.0)
                intercept = getattr(ds, "RescaleIntercept", 0.0)
                hu = pixels * slope + intercept

                # Apply windowing
                lower = window_center - (window_width / 2.0)
                upper = window_center + (window_width / 2.0)
                windowed = np.clip((hu - lower) / (upper - lower), 0.0, 1.0)
                yield fname, windowed
            except Exception as e:
                logger.debug("Failed reading DICOM %s: %s", fpath, e)
