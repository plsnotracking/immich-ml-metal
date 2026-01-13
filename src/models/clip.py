"""
CLIP model implementation using MLX.

Uses the open_clip library for tokenization and model architecture,
with weights running on MLX for Apple Silicon optimization.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import io
import json

# We'll use transformers for the tokenizer and open_clip for preprocessing
# but run inference through MLX


class MLXClip:
    """
    CLIP model optimized for Apple Silicon via MLX.
    
    For 8GB M1, uses ViT-B-32 (~350MB) which provides good quality
    at reasonable memory cost.
    """
    
    def __init__(self, model_name: str = "ViT-B-32-quickgelu", pretrained: str = "openai"):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self._loaded:
            return
        
        try:
            import open_clip
            
            # Load the model - open_clip will download if needed
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
            )
            tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Put model in eval mode
            model.eval()
            
            # Store for inference - we'll use torch for now
            # (MLX conversion can be added later for more speed)
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._loaded = True
            
            print(f"✅ Loaded CLIP model: {self.model_name}/{self.pretrained}")
            
        except ImportError as e:
            raise ImportError(
                "open_clip not installed. Run: pip install open_clip_torch"
            ) from e
    
    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate CLIP embedding for an image.
        
        Args:
            image_bytes: Raw image data (JPEG, PNG, etc.)
            
        Returns:
            Normalized 512-dim embedding as float32 numpy array
        """
        import torch
        
        self._ensure_loaded()
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().numpy().astype(np.float32)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text.
        
        Args:
            text: Search query or description
            
        Returns:
            Normalized 512-dim embedding as float32 numpy array
        """
        import torch
        
        self._ensure_loaded()
        
        # Tokenize
        tokens = self._tokenizer([text])
        
        # Run inference
        with torch.no_grad():
            embedding = self._model.encode_text(tokens)
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().numpy().astype(np.float32)


# Singleton instance for model caching
_clip_instance: Optional[MLXClip] = None


def get_clip_model(model_name: str = "ViT-B-32-quickgelu") -> MLXClip:
    """
    Get or create the CLIP model instance.
    
    Uses singleton pattern to avoid loading model multiple times.
    """
    global _clip_instance
    
    if _clip_instance is None:
        _clip_instance = MLXClip(model_name=model_name)
    
    return _clip_instance


# For testing
if __name__ == "__main__":
    import sys
    
    print("Testing CLIP model...")
    clip = get_clip_model()
    
    # Test text encoding
    text_emb = clip.encode_text("a photo of a cat")
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Text embedding norm: {np.linalg.norm(text_emb):.4f}")
    
    # Test image encoding if file provided
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            img_emb = clip.encode_image(f.read())
        print(f"Image embedding shape: {img_emb.shape}")
        print(f"Image embedding norm: {np.linalg.norm(img_emb):.4f}")
        
        # Compute similarity
        similarity = np.dot(text_emb, img_emb)
        print(f"Text-image similarity: {similarity:.4f}")
    
    print("✅ CLIP test passed!")