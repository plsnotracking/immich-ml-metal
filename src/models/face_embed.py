"""
Face embedding generation using InsightFace ArcFace model.

Uses CoreML execution provider when available for Apple Silicon acceleration.
Thread-safe for both loading and inference.
"""

import numpy as np
from PIL import Image
import cv2
import io
from typing import Optional
from pathlib import Path
import logging
import threading
import gc

logger = logging.getLogger(__name__)

# Global model cache with thread safety
_recognition_model = None
_current_model_name: Optional[str] = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()  # Separate lock for inference


def get_recognition_model(model_name: str = "buffalo_l"):
    """
    Get or create the face recognition model (thread-safe).
    
    Loads the recognition model directly from insightface model zoo.
    If a different model is requested, unloads the current one first.
    
    Args:
        model_name: InsightFace model name (buffalo_s, buffalo_m, buffalo_l)
    """
    global _recognition_model, _current_model_name
    
    with _model_lock:
        # Check if we need to switch models
        if _recognition_model is not None and _current_model_name != model_name:
            logger.info(f"Switching face model: {_current_model_name} -> {model_name}")
            unload_recognition_model()
        
        # Load model if needed
        if _recognition_model is None:
            logger.info(f"Loading face recognition model: {model_name}")
            _load_model(model_name)
            _current_model_name = model_name
        
        return _recognition_model


def _load_model(model_name: str):
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
    
    # Check available providers
    available = ort.get_available_providers()
    logger.info(f"ONNX Runtime providers available: {available}")
    
    # Prefer CoreML for Apple Silicon
    if "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CoreML for face recognition")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("Using CPU for face recognition (CoreML not available)")
    
    # Download model pack if needed
    model_dir = Path.home() / ".insightface" / "models" / model_name
    if not model_dir.exists():
        logger.info(f"Downloading {model_name} model pack (first run, ~60-350MB)")
        try:
            ensure_available("models", model_name, root=str(Path.home() / ".insightface"))
        except Exception as e:
            logger.error(f"Failed to download model pack: {e}", exc_info=True)
            raise RuntimeError(f"Could not download {model_name} model pack") from e
    
    # Find the recognition model file
    rec_model_path = None
    for f in model_dir.glob("*.onnx"):
        # Recognition models have w600k, w300k, or glintr in name
        if any(pattern in f.name.lower() for pattern in ["w600k", "w300k", "glintr"]):
            rec_model_path = f
            break
    
    if rec_model_path is None:
        logger.error(f"No recognition model found in {model_dir}")
        logger.info(f"Available files: {list(model_dir.glob('*.onnx'))}")
        raise FileNotFoundError(f"No recognition model found in {model_dir}")
    
    logger.info(f"Loading recognition model: {rec_model_path.name}")
    
    try:
        _recognition_model = model_zoo.get_model(str(rec_model_path), providers=providers)
        logger.info(f"Successfully loaded face recognition model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load recognition model: {e}", exc_info=True)
        raise RuntimeError(f"Could not load {model_name} recognition model") from e


def unload_recognition_model():
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
    model_name: str = "buffalo_l"
) -> np.ndarray:
    """
    Generate 512-dim face embedding using ArcFace (thread-safe).
    
    Args:
        image_bytes: Raw image data
        landmarks: 5-point landmarks [[x,y], ...] from Vision framework
                  Order: left_eye, right_eye, nose, left_mouth, right_mouth
        model_name: InsightFace model to use
        
    Returns:
        512-dimensional normalized embedding as float32 array
    """
    try:
        from insightface.utils import face_align
    except ImportError as e:
        logger.error("insightface not available")
        raise RuntimeError("Install insightface: pip install insightface") from e
    
    # Decode image to BGR (OpenCV format)
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise
    
    # Convert landmarks to numpy array
    kps = np.array(landmarks, dtype=np.float32)
    
    # Align face using landmarks (ArcFace expects 112x112 aligned face)
    try:
        aligned_face = face_align.norm_crop(img_bgr, kps, image_size=112)
    except Exception as e:
        logger.error(f"Face alignment failed: {e}")
        raise
    
    # Get recognition model
    model = get_recognition_model(model_name)
    
    # CRITICAL: Lock inference to prevent concurrent operations
    # While ONNX Runtime with CoreML is more robust than MLX,
    # we still serialize to be safe and avoid resource contention
    with _inference_lock:
        try:
            embedding = model.get_feat(aligned_face)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    # Normalize
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype(np.float32)


def get_face_embedding_from_bbox(
    image_bytes: bytes,
    bbox: dict,
    model_name: str = "buffalo_l"
) -> Optional[np.ndarray]:
    """
    Generate face embedding using bounding box (fallback, thread-safe).
    
    Less accurate than landmark-based alignment but works as fallback.
    
    Args:
        image_bytes: Raw image data
        bbox: Bounding box dict with x1, y1, x2, y2
        model_name: InsightFace model to use
        
    Returns:
        512-dimensional normalized embedding, or None if failed
    """
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.error("Failed to decode image")
            return None
        
        # Crop face region with padding
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
        
        # Resize to 112x112 (ArcFace input size)
        face_resized = cv2.resize(face_crop, (112, 112))
        
        # Get embedding
        model = get_recognition_model(model_name)
        
        # CRITICAL: Lock inference
        with _inference_lock:
            embedding = model.get_feat(face_resized)
        
        # Normalize
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Embedding from bbox failed: {e}", exc_info=True)
        return None


# Alias for backward compatibility
def get_face_recognizer(model_name: str = "buffalo_l"):
    """Alias for get_recognition_model for backward compatibility."""
    return get_recognition_model(model_name)


if __name__ == "__main__":
    import sys
    
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger.info("Testing face embedding generation...")
    logger.info("(This will download the model on first run, ~60-350MB depending on model)")
    
    # Test model loading
    model = get_recognition_model("buffalo_l")
    logger.info(f"Model loaded successfully: {type(model).__name__}")
    
    # If image provided, test full pipeline
    if len(sys.argv) > 1:
        from .face_detect import detect_faces
        
        with open(sys.argv[1], "rb") as f:
            image_bytes = f.read()
        
        faces, w, h = detect_faces(image_bytes)
        logger.info(f"Detected {len(faces)} face(s) in {w}x{h} image")
        
        for i, face in enumerate(faces):
            logger.info(f"\nFace {i + 1}:")
            logger.info(f"  BBox: ({face['boundingBox']['x1']:.0f}, {face['boundingBox']['y1']:.0f}) - "
                       f"({face['boundingBox']['x2']:.0f}, {face['boundingBox']['y2']:.0f})")
            logger.info(f"  Score: {face['score']:.3f}")
            
            if "landmarks" in face:
                embedding = get_face_embedding(image_bytes, face["landmarks"], "buffalo_l")
                logger.info(f"  Embedding (landmark-aligned): {embedding.shape}")
            else:
                embedding = get_face_embedding_from_bbox(image_bytes, face["boundingBox"], "buffalo_l")
                logger.info(f"  Embedding (bbox-cropped): {embedding.shape}")
            
            if embedding is not None:
                logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
                logger.info(f"  First 5 values: {embedding[:5]}")
    
    logger.info("\n✅ Face embedding test complete!")