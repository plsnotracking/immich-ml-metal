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
    # buffalo_s (~30MB) vs buffalo_l (~350MB) - much smaller!
    face_model: str = "buffalo_s"
    face_min_score: float = 0.034  # Immich default from locustfile
    
    # OCR settings
    ocr_min_detection_score: float = 0.5
    ocr_min_recognition_score: float = 0.5
    ocr_max_resolution: int = 1024
    
    # Performance settings
    use_coreml: bool = True
    use_ane: bool = True  # Apple Neural Engine
    
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
        )


# Global settings instance
settings = Settings.from_env()