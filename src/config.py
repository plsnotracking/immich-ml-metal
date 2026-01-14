"""Configuration settings for immich-ml-metal."""

import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Settings:
    """Application settings with sensible defaults."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 3003
    
    # Model paths
    models_dir: Path = field(default_factory=lambda: Path("./models"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    
    # CLIP settings - using smaller model for low-memory systems
    # ViT-B-32 (~350MB) vs ViT-L-14 (~1.7GB)
    clip_model: str = "ViT-B-32__openai"
    clip_model_path: Path = field(default_factory=lambda: Path("./models/clip-vit-base-patch32-mlx"))
    
    # Face recognition settings
    # buffalo_l is Immich's default, provides best accuracy
    # buffalo_s (~60MB total) and buffalo_m (~150MB) are smaller alternatives
    face_model: str = "buffalo_l"
    
    # Face detection threshold - Immich default is 0.7
    # Lower values = more faces detected (more false positives)
    # Higher values = fewer faces detected (more false negatives)
    face_min_score: float = 0.7
    
    # OCR settings
    ocr_min_detection_score: float = 0.5
    ocr_min_recognition_score: float = 0.5
    ocr_max_resolution: int = 1024
    
    # Performance settings
    use_coreml: bool = True
    use_ane: bool = True  # Apple Neural Engine
    
    # Resource limits
    max_image_size: int = 50 * 1024 * 1024  # 50MB max upload
    request_timeout: int = 120  # 2 minutes max for ML inference
    
    def __post_init__(self):
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            host=os.getenv("ML_HOST", "0.0.0.0"),
            port=int(os.getenv("ML_PORT", "3003")),
            models_dir=Path(os.getenv("ML_MODELS_DIR", "./models")),
            cache_dir=Path(os.getenv("ML_CACHE_DIR", "./cache")),
            clip_model=os.getenv("ML_CLIP_MODEL", "ViT-B-32__openai"),
            face_model=os.getenv("ML_FACE_MODEL", "buffalo_l"),
            face_min_score=float(os.getenv("ML_FACE_MIN_SCORE", "0.7")),
            use_coreml=os.getenv("ML_USE_COREML", "true").lower() == "true",
            use_ane=os.getenv("ML_USE_ANE", "true").lower() == "true",
            max_image_size=int(os.getenv("ML_MAX_IMAGE_SIZE", str(50 * 1024 * 1024))),
            request_timeout=int(os.getenv("ML_REQUEST_TIMEOUT", "120")),
        )


# Global settings instance
settings = Settings.from_env()