"""Configuration settings for immich-ml-metal."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias

LogLevel: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Supported face recognition models
SUPPORTED_FACE_MODELS: Final[tuple[str, ...]] = (
    "buffalo_s",   # Small: ~60MB, fastest, lower accuracy
    "buffalo_m",   # Medium: ~150MB, balanced
    "buffalo_l",   # Large: ~350MB, best accuracy (Immich default)
    "antelopev2",  # Alternative: ~200MB, high-quality recognition
)


@dataclass(slots=True)
class Settings:
    """Application settings with sensible defaults."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 3003
    
    # Model paths
    models_dir: Path = field(default_factory=lambda: Path("./models"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    
    # CLIP settings - using smaller model for low-memory systems
    clip_model: str = "ViT-B-32__openai"
    clip_model_path: Path = field(default_factory=lambda: Path("./models/clip-vit-base-patch32-mlx"))
    
    # CLIP image buffer settings
    # Images are buffered in RAM before encoding; overflow to disk if limit exceeded
    clip_buffer_ram_limit_mb: int = 256  # Max MB to buffer in RAM for encode queue
    
    # Face recognition settings
    # Supported models: buffalo_s, buffalo_m, buffalo_l, antelopev2
    # - buffalo_l: Immich's default, best accuracy (~350MB)
    # - buffalo_m: Balanced option (~150MB)
    # - buffalo_s: Smallest/fastest (~60MB)
    # - antelopev2: High-quality alternative (~200MB)
    face_model: str = "buffalo_l"
    
    # Face detection threshold - Immich default is 0.7
    # Lower values = more faces detected (more false positives)
    # Higher values = fewer faces detected (more false negatives)
    face_min_score: float = 0.7
    
    # OCR settings
    ocr_min_detection_score: float = 0.5
    ocr_min_recognition_score: float = 0.5
    ocr_max_resolution: int = 1024
    ocr_use_language_correction: bool = True  # Disable for technical text/codes
    
    # Performance settings
    use_coreml: bool = True
    use_ane: bool = True  # Apple Neural Engine
    max_concurrent_requests: int = 4  # Queued requests before backpressure
    
    # Resource limits
    max_image_size: int = 50 * 1024 * 1024  # 50MB max upload
    request_timeout: int = 120  # 2 minutes max for ML inference
    
    # Logging settings
    log_level: LogLevel = "INFO"
    log_requests: bool = True  # Log individual requests (disable for high volume)
    
    # Debug mode - when True, expose error details in responses
    # Should be False when service is network-accessible
    debug_mode: bool = False
    
    def __post_init__(self):
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables."""
        return cls(
            host=os.getenv("ML_HOST", "0.0.0.0"),
            port=int(os.getenv("ML_PORT", "3003")),
            models_dir=Path(os.getenv("ML_MODELS_DIR", "./models")),
            cache_dir=Path(os.getenv("ML_CACHE_DIR", "./cache")),
            clip_model=os.getenv("ML_CLIP_MODEL", "ViT-B-32__openai"),
            clip_buffer_ram_limit_mb=int(os.getenv("ML_CLIP_BUFFER_RAM_MB", "256")),
            face_model=os.getenv("ML_FACE_MODEL", "buffalo_l"),
            face_min_score=float(os.getenv("ML_FACE_MIN_SCORE", "0.7")),
            ocr_use_language_correction=os.getenv("ML_OCR_LANGUAGE_CORRECTION", "true").lower() == "true",
            use_coreml=os.getenv("ML_USE_COREML", "true").lower() == "true",
            use_ane=os.getenv("ML_USE_ANE", "true").lower() == "true",
            max_concurrent_requests=int(os.getenv("ML_MAX_CONCURRENT_REQUESTS", "4")),
            max_image_size=int(os.getenv("ML_MAX_IMAGE_SIZE", str(50 * 1024 * 1024))),
            request_timeout=int(os.getenv("ML_REQUEST_TIMEOUT", "120")),
            log_level=os.getenv("ML_LOG_LEVEL", "INFO").upper(),  # type: ignore[arg-type]
            log_requests=os.getenv("ML_LOG_REQUESTS", "true").lower() == "true",
            debug_mode=os.getenv("ML_DEBUG_MODE", "false").lower() == "true",
        )
    
    def configure_logging(self):
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


# Global settings instance
settings = Settings.from_env()
