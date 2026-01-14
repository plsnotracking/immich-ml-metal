"""
Model implementations for immich-ml-metal.

Modules:
    - clip: CLIP image/text embeddings (MLX/open_clip)
    - face_detect: Face detection (Apple Vision framework)
    - face_embed: Face embeddings (InsightFace ArcFace)
    - ocr: Text recognition (Apple Vision framework)

Supported face models:
    - buffalo_s, buffalo_m, buffalo_l (InsightFace buffalo series)
    - antelopev2 (high-quality alternative)
"""

from __future__ import annotations

from .clip import MLXClip, get_clip_model
from .face_detect import detect_faces
from .face_embed import (
    get_face_embedding,
    get_face_embedding_from_bbox,
    get_recognition_model,
    get_supported_models,
)
from .ocr import recognize_text

__all__ = [
    # CLIP
    "get_clip_model",
    "MLXClip",
    # Face detection
    "detect_faces",
    # Face embedding
    "get_face_embedding",
    "get_face_embedding_from_bbox",
    "get_recognition_model",
    "get_supported_models",
    # OCR
    "recognize_text",
]
