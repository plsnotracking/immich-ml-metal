"""
Face embedding generation using InsightFace ArcFace model.

Uses CoreML execution provider when available for Apple Silicon acceleration.
Loads recognition model directly (not via FaceAnalysis) since we use
Vision framework for detection.
"""

import numpy as np
from PIL import Image
import cv2
import io
from typing import Optional
from pathlib import Path

# Global model cache
_recognition_model = None


def get_recognition_model(model_name: str = "buffalo_s"):
    """
    Get or create the face recognition model.
    
    Loads the recognition model directly from insightface model zoo,
    bypassing FaceAnalysis which requires detection.
    
    Args:
        model_name: InsightFace model name (buffalo_s, buffalo_m, buffalo_l)
    """
    global _recognition_model
    
    if _recognition_model is not None:
        return _recognition_model
    
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo
    from insightface.utils import DEFAULT_MP_NAME, ensure_available
    
    # Check available providers
    available = ort.get_available_providers()
    print(f"ONNX Runtime providers: {available}")
    
    # Prefer CoreML, fall back to CPU
    if "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        print("Using CoreML for face recognition")
    else:
        providers = ["CPUExecutionProvider"]
        print("Using CPU for face recognition (CoreML not available)")
    
    # Download model pack if needed
    model_dir = Path.home() / ".insightface" / "models" / model_name
    if not model_dir.exists():
        # This will download the model pack
        ensure_available("models", model_name, root=str(Path.home() / ".insightface"))
    
    # Find the recognition model file (w600k_*.onnx or similar)
    rec_model_path = None
    for f in model_dir.glob("*.onnx"):
        # Recognition models typically have 'w600k' or 'w300k' in name
        if "w600k" in f.name or "w300k" in f.name or "glintr" in f.name:
            rec_model_path = f
            break
    
    if rec_model_path is None:
        raise FileNotFoundError(f"No recognition model found in {model_dir}")
    
    print(f"Loading recognition model: {rec_model_path.name}")
    
    # Load model directly using model_zoo
    _recognition_model = model_zoo.get_model(str(rec_model_path), providers=providers)
    
    print(f"✅ Loaded face recognition model: {model_name}")
    return _recognition_model


def get_face_embedding(
    image_bytes: bytes,
    landmarks: list[list[float]],
    model_name: str = "buffalo_s"
) -> np.ndarray:
    """
    Generate 512-dim face embedding using ArcFace.
    
    Args:
        image_bytes: Raw image data
        landmarks: 5-point landmarks [[x,y], ...] from Vision framework
                  Order: left_eye, right_eye, nose, left_mouth, right_mouth
        model_name: InsightFace model to use
        
    Returns:
        512-dimensional normalized embedding as float32 array
    """
    from insightface.utils import face_align
    
    # Decode image to BGR (OpenCV format)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert landmarks to numpy array
    kps = np.array(landmarks, dtype=np.float32)
    
    # Align face using landmarks (ArcFace expects 112x112 aligned face)
    aligned_face = face_align.norm_crop(img_bgr, kps, image_size=112)
    
    # Get recognition model
    model = get_recognition_model(model_name)
    
    # Get embedding - use get_feat() which takes the aligned face directly
    embedding = model.get_feat(aligned_face)
    
    # Normalize
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype(np.float32)


def get_face_embedding_from_bbox(
    image_bytes: bytes,
    bbox: dict,
    model_name: str = "buffalo_s"
) -> Optional[np.ndarray]:
    """
    Generate face embedding using bounding box (when landmarks aren't available).
    
    This is less accurate than landmark-based alignment but works as fallback.
    
    Args:
        image_bytes: Raw image data
        bbox: Bounding box dict with x1, y1, x2, y2
        model_name: InsightFace model to use
        
    Returns:
        512-dimensional normalized embedding, or None if failed
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Crop face region with some padding
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
    
    # Resize to 112x112 (ArcFace input size)
    face_resized = cv2.resize(face_crop, (112, 112))
    
    # Get embedding
    model = get_recognition_model(model_name)
    embedding = model.get_feat(face_resized)
    
    # Normalize
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype(np.float32)


# Alias for compatibility
def get_face_recognizer(model_name: str = "buffalo_s"):
    """Alias for get_recognition_model for backward compatibility."""
    return get_recognition_model(model_name)


# For testing
if __name__ == "__main__":
    import sys
    
    print("Testing face embedding generation...")
    print("(This will download the model on first run, ~30MB for buffalo_s)\n")
    
    # Test that model loads
    model = get_recognition_model("buffalo_s")
    print(f"Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    
    # If image provided, try full pipeline
    if len(sys.argv) > 1:
        from .face_detect import detect_faces
        
        with open(sys.argv[1], "rb") as f:
            image_bytes = f.read()
        
        # Detect faces
        faces, w, h = detect_faces(image_bytes)
        print(f"\nDetected {len(faces)} face(s) in {w}x{h} image")
        
        for i, face in enumerate(faces):
            print(f"\nFace {i + 1}:")
            print(f"  BBox: ({face['boundingBox']['x1']:.0f}, {face['boundingBox']['y1']:.0f}) - ({face['boundingBox']['x2']:.0f}, {face['boundingBox']['y2']:.0f})")
            print(f"  Score: {face['score']:.3f}")
            
            if "landmarks" in face:
                embedding = get_face_embedding(
                    image_bytes, 
                    face["landmarks"],
                    "buffalo_s"
                )
                print(f"  Embedding (landmark-aligned): {embedding.shape}")
            else:
                embedding = get_face_embedding_from_bbox(
                    image_bytes,
                    face["boundingBox"],
                    "buffalo_s"
                )
                print(f"  Embedding (bbox-cropped): {embedding.shape}")
            
            print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"  First 5 values: {embedding[:5]}")
    
    print("\n✅ Face embedding test complete!")