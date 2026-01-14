"""
Face embedding generation using InsightFace ArcFace model.

Uses CoreML execution provider when available for Apple Silicon acceleration.
Thread-safe for both loading and inference.

Supported models:
    - buffalo_s: Small model (~60MB), fastest, lower accuracy
    - buffalo_m: Medium model (~150MB), balanced
    - buffalo_l: Large model (~350MB), best accuracy (default)
    - antelopev2: High-quality model with improved recognition
"""

from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Final, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Expected model characteristics for ArcFace recognition models
ARCFACE_INPUT_SIZE: Final[int] = 112
ARCFACE_EMBEDDING_DIM: Final[int] = 512

# Supported model configurations with their recognition model patterns
MODEL_CONFIGS: Final[dict[str, list[str]]] = {
    "buffalo_s": ["w600k", "w300k", "glintr"],
    "buffalo_m": ["w600k", "w300k", "glintr"],
    "buffalo_l": ["w600k", "w300k", "glintr"],
    "antelopev2": ["glintr100", "glintr", "w600k"],
}

# Global model cache with thread safety
_recognition_model: object | None = None
_current_model_name: str | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_recognition_model(model_name: str = "buffalo_l") -> object:
    """
    Get or create the face recognition model (thread-safe).
    
    Loads the recognition model directly from insightface model zoo.
    If a different model is requested, unloads the current one first.
    
    Args:
        model_name: InsightFace model name (buffalo_s/m/l, antelopev2)
        
    Returns:
        The loaded recognition model instance.
        
    Raises:
        RuntimeError: If model cannot be loaded.
        ValueError: If model_name is not supported.
    """
    global _recognition_model, _current_model_name
    
    # Validate model name
    if model_name not in MODEL_CONFIGS:
        supported = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unsupported model: {model_name}. Supported: {supported}")
    
    with _model_lock:
        if _recognition_model is not None and _current_model_name != model_name:
            logger.info(f"Switching face model: {_current_model_name} -> {model_name}")
            unload_recognition_model()
        
        if _recognition_model is None:
            logger.info(f"Loading face recognition model: {model_name}")
            _load_model(model_name)
            _current_model_name = model_name
        
        return _recognition_model


def _find_recognition_model(model_dir: Path) -> Optional[Path]:
    """
    Find the recognition model in a model pack directory.
    
    Uses shape validation to identify ArcFace models rather than
    relying on filename patterns which may change between versions.
    
    Returns:
        Path to the recognition model, or None if not found.
    """
    import onnxruntime as ort
    
    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        return None
    
    # Known filename patterns (checked first for speed)
    known_patterns = ["w600k", "w300k", "glintr", "arcface"]
    
    for f in onnx_files:
        if any(pattern in f.name.lower() for pattern in known_patterns):
            # Verify it's actually a recognition model by checking shapes
            if _validate_recognition_model(f):
                return f
    
    # Fallback: check all ONNX files for correct input/output shapes
    for f in onnx_files:
        if _validate_recognition_model(f):
            logger.info(f"Found recognition model by shape validation: {f.name}")
            return f
    
    return None


def _validate_recognition_model(model_path: Path) -> bool:
    """
    Validate that an ONNX model is an ArcFace recognition model.
    
    Checks:
    - Input shape compatible with (batch, 3, 112, 112)
    - Output shape compatible with (batch, 512)
    """
    try:
        import onnxruntime as ort
        
        # Quick session just to check metadata
        sess = ort.InferenceSession(
            str(model_path), 
            providers=["CPUExecutionProvider"]
        )
        
        # Check input shape
        input_info = sess.get_inputs()[0]
        input_shape = input_info.shape
        # Shape could be [batch, 3, 112, 112] or with dynamic batch
        if len(input_shape) != 4:
            return False
        # Check spatial dimensions (indices 2 and 3)
        if input_shape[2] != ARCFACE_INPUT_SIZE or input_shape[3] != ARCFACE_INPUT_SIZE:
            return False
        # Check channels (index 1)
        if input_shape[1] != 3:
            return False
        
        # Check output shape
        output_info = sess.get_outputs()[0]
        output_shape = output_info.shape
        if len(output_shape) != 2:
            return False
        # Check embedding dimension (index 1)
        if output_shape[1] != ARCFACE_EMBEDDING_DIM:
            return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Model validation failed for {model_path}: {e}")
        return False


def _load_model(model_name: str) -> None:
    """Internal function to load the model (assumes lock is held)."""
    global _recognition_model
    
    try:
        import onnxruntime as ort
        from insightface.model_zoo import model_zoo
        from insightface.utils import ensure_available
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        raise RuntimeError(
            "insightface or onnxruntime not available. "
            "Install with: pip install insightface onnxruntime"
        ) from e
    
    available = ort.get_available_providers()
    logger.info(f"ONNX Runtime providers available: {available}")
    
    # Prefer CoreML for Apple Silicon
    match "CoreMLExecutionProvider" in available:
        case True:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CoreML for face recognition")
        case False:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU for face recognition (CoreML not available)")
    
    # Download model pack if needed
    base_model_dir = Path.home() / ".insightface" / "models" / model_name
    if not base_model_dir.exists():
        logger.info(f"Downloading {model_name} model pack (first run)")
        try:
            ensure_available("models", model_name, root=str(Path.home() / ".insightface"))
        except Exception as e:
            logger.error(f"Failed to download model pack: {e}", exc_info=True)
            raise RuntimeError(f"Could not download {model_name} model pack") from e
    
    # Handle nested directory structure (some models extract to model_name/model_name/)
    model_dir = base_model_dir
    nested_dir = base_model_dir / model_name
    if nested_dir.exists() and any(nested_dir.glob("*.onnx")):
        model_dir = nested_dir
        logger.debug(f"Using nested model directory: {model_dir}")
    
    # Find the recognition model using robust detection
    rec_model_path = _find_recognition_model(model_dir)
    
    # Also check base directory if nested didn't have the model
    if rec_model_path is None and model_dir != base_model_dir:
        rec_model_path = _find_recognition_model(base_model_dir)
    
    if rec_model_path is None:
        available_files = [f.name for f in model_dir.glob("*.onnx")]
        if model_dir != base_model_dir:
            available_files.extend([f.name for f in base_model_dir.glob("*.onnx")])
        logger.error(f"No valid recognition model found in {model_dir}")
        logger.error(f"Available ONNX files: {available_files}")
        raise FileNotFoundError(
            f"No ArcFace recognition model (input: 3x112x112, output: 512-dim) "
            f"found in {model_dir}"
        )
    
    logger.info(f"Loading recognition model: {rec_model_path.name}")
    
    try:
        _recognition_model = model_zoo.get_model(str(rec_model_path), providers=providers)
        logger.info(f"Successfully loaded face recognition model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load recognition model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load {model_name} recognition model") from e


def unload_recognition_model() -> None:
    """Unload the current recognition model and free memory."""
    global _recognition_model, _current_model_name
    
    if _recognition_model is not None:
        logger.info(f"Unloading face recognition model: {_current_model_name}")
        _recognition_model = None
        _current_model_name = None
        gc.collect()


def get_face_embedding(
    image_bytes: bytes,
    landmarks: list[list[float]],
    model_name: str = "buffalo_l",
) -> NDArray[np.float32]:
    """
    Generate 512-dim face embedding using ArcFace (thread-safe).
    
    Args:
        image_bytes: Raw image data.
        landmarks: 5-point landmarks [[x,y], ...] from Vision framework.
                   Order: left_eye, right_eye, nose, left_mouth, right_mouth.
        model_name: InsightFace model to use.
        
    Returns:
        512-dimensional normalized embedding as float32 array.
        
    Raises:
        RuntimeError: If insightface is not available.
        ValueError: If image decoding fails.
    """
    try:
        from insightface.utils import face_align
    except ImportError as e:
        logger.error("insightface not available")
        raise RuntimeError("Install insightface: pip install insightface") from e
    
    # Decode image to BGR (OpenCV format)
    img_bgr = _decode_image(image_bytes)
    
    kps = np.array(landmarks, dtype=np.float32)
    
    try:
        aligned_face = face_align.norm_crop(img_bgr, kps, image_size=ARCFACE_INPUT_SIZE)
    except Exception as e:
        logger.error(f"Face alignment failed: {e}")
        raise
    
    model = get_recognition_model(model_name)
    
    with _inference_lock:
        try:
            embedding = model.get_feat(aligned_face)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    return _normalize_embedding(embedding)


def get_face_embedding_from_bbox(
    image_bytes: bytes,
    bbox: dict[str, int],
    model_name: str = "buffalo_l",
) -> NDArray[np.float32] | None:
    """
    Generate face embedding using bounding box (fallback, thread-safe).
    
    Less accurate than landmark-based alignment but works as fallback.
    
    Args:
        image_bytes: Raw image data.
        bbox: Bounding box dict with x1, y1, x2, y2.
        model_name: InsightFace model to use.
        
    Returns:
        512-dimensional normalized embedding, or None if failed.
    """
    try:
        # Decode image
        img_bgr = _decode_image(image_bytes)
        
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        
        # Add 10% padding
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_bgr.shape[1], x2 + pad_x)
        y2 = min(img_bgr.shape[0], y2 + pad_y)
        
        face_crop = img_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            logger.error("Empty face crop")
            return None
        
        face_resized = cv2.resize(face_crop, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE))
        
        model = get_recognition_model(model_name)
        
        with _inference_lock:
            embedding = model.get_feat(face_resized)
        
        return _normalize_embedding(embedding)
        
    except Exception as e:
        logger.error(f"Embedding from bbox failed: {e}", exc_info=True)
        return None


def _decode_image(image_bytes: bytes) -> NDArray[np.uint8]:
    """
    Decode image bytes to BGR format for OpenCV.
    
    Args:
        image_bytes: Raw image data.
        
    Returns:
        Image as BGR numpy array.
        
    Raises:
        ValueError: If image cannot be decoded.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image")
        return img_bgr
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise


def _normalize_embedding(embedding: NDArray) -> NDArray[np.float32]:
    """
    Normalize embedding to unit vector.
    
    Args:
        embedding: Raw embedding from model.
        
    Returns:
        Normalized float32 embedding.
    """
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)


# Alias for backward compatibility
def get_face_recognizer(model_name: str = "buffalo_l") -> object:
    """Alias for get_recognition_model for backward compatibility."""
    return get_recognition_model(model_name)


def get_supported_models() -> list[str]:
    """Return list of supported face recognition model names."""
    return list(MODEL_CONFIGS.keys())


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Parse optional model argument
    test_model = sys.argv[2] if len(sys.argv) > 2 else "buffalo_l"
    
    logger.info("Testing face embedding generation...")
    logger.info(f"Supported models: {get_supported_models()}")
    logger.info(f"Testing model: {test_model}")
    logger.info("(This will download the model on first run)")
    
    # Test model loading
    try:
        model = get_recognition_model(test_model)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
    except ValueError as e:
        logger.error(f"Invalid model: {e}")
        sys.exit(1)
    
    # If image provided, test full pipeline
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        from .face_detect import detect_faces
        
        image_path = Path(sys.argv[1])
        image_bytes = image_path.read_bytes()
        
        faces, w, h = detect_faces(image_bytes)
        logger.info(f"Detected {len(faces)} face(s) in {w}x{h} image")
        
        for i, face in enumerate(faces):
            logger.info(f"\nFace {i + 1}:")
            bbox = face["boundingBox"]
            logger.info(
                f"  BBox: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - "
                f"({bbox['x2']:.0f}, {bbox['y2']:.0f})"
            )
            logger.info(f"  Score: {face['score']:.3f}")
            
            if "landmarks" in face:
                embedding = get_face_embedding(image_bytes, face["landmarks"], test_model)
                logger.info(f"  Embedding (landmark-aligned): {embedding.shape}")
            else:
                embedding = get_face_embedding_from_bbox(image_bytes, face["boundingBox"], test_model)
                logger.info(f"  Embedding (bbox-cropped): {embedding.shape if embedding is not None else 'failed'}")
            
            if embedding is not None:
                logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
                logger.info(f"  First 5 values: {embedding[:5]}")
    
    logger.info("\n✅ Face embedding test complete!")
