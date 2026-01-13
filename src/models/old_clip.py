"""
CLIP model implementation using MLX for Apple Silicon acceleration.

Supports dynamic model loading based on Immich requests.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import io
import gc

# Model name mapping: Immich name -> HuggingFace mlx-community repo
# Immich uses format like "ViT-B-32__openai" or "ViT-B-32::openai"
MODEL_MAP = {
    # OpenAI models
    "ViT-B-32__openai": "mlx-community/clip-vit-base-patch32",
    "ViT-B-32::openai": "mlx-community/clip-vit-base-patch32",
    "ViT-B-16__openai": "mlx-community/clip-vit-base-patch16",
    "ViT-B-16::openai": "mlx-community/clip-vit-base-patch16",
    "ViT-L-14__openai": "mlx-community/clip-vit-large-patch14",
    "ViT-L-14::openai": "mlx-community/clip-vit-large-patch14",
    
    # LAION models
    "ViT-B-32__laion2b_s34b_b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    "ViT-B-32::laion2b_s34b_b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    
    # Default fallback
    "default": "mlx-community/clip-vit-base-patch32",
}


class MLXClip:
    """
    CLIP model using MLX for Apple Silicon acceleration.
    
    Uses mlx-community models from HuggingFace.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False
        
        # Resolve model repo
        self._repo_id = MODEL_MAP.get(model_name, MODEL_MAP.get("default"))
        
        self._load_model()
    
    def _load_model(self):
        """Load the MLX CLIP model."""
        try:
            from mlx_clip import mlx_clip
            
            print(f"Loading MLX CLIP model: {self.model_name} -> {self._repo_id}")
            
            # mlx_clip handles downloading and loading
            self._model = mlx_clip(self._repo_id)
            self._loaded = True
            
            print(f"✅ Loaded CLIP model: {self.model_name}")
            
        except ImportError:
            # Fallback to open_clip with MPS if mlx_clip not available
            print("⚠️ mlx_clip not available, falling back to open_clip with MPS")
            self._load_fallback()
    
    def _load_fallback(self):
        """Fallback to open_clip with MPS acceleration."""
        import torch
        import open_clip
        
        # Parse model name
        if "__" in self.model_name:
            arch, pretrained = self.model_name.split("__", 1)
        elif "::" in self.model_name:
            arch, pretrained = self.model_name.split("::", 1)
        else:
            arch = "ViT-B-32"
            pretrained = "openai"
        
        # OpenAI models need quickgelu variant
        if pretrained == "openai" and "quickgelu" not in arch.lower():
            arch = arch + "-quickgelu"
        
        print(f"Loading open_clip model: {arch}/{pretrained}")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(arch)
        
        # Use MPS if available
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            model = model.to(self._device)
            print("✅ Using MPS (Metal) acceleration")
        else:
            self._device = torch.device("cpu")
            print("⚠️ MPS not available, using CPU")
        
        model.eval()
        
        self._model = model
        self._processor = preprocess
        self._tokenizer = tokenizer
        self._use_fallback = True
        self._loaded = True
        
        print(f"✅ Loaded CLIP model: {arch}/{pretrained}")
    
    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Generate CLIP embedding for an image."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if hasattr(self, '_use_fallback') and self._use_fallback:
            return self._encode_image_fallback(image)
        
        # MLX path
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f, format="JPEG")
            temp_path = f.name
        
        try:
            embedding = self._model.image_encoder(temp_path)
            # Normalize
            if isinstance(embedding, mx.array):
                embedding = np.array(embedding)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.flatten().astype(np.float32)
        finally:
            Path(temp_path).unlink()
    
    def _encode_image_fallback(self, image: Image.Image) -> np.ndarray:
        """Encode image using open_clip fallback."""
        import torch
        
        image_tensor = self._processor(image).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().numpy().astype(np.float32)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Generate CLIP embedding for text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        if hasattr(self, '_use_fallback') and self._use_fallback:
            return self._encode_text_fallback(text)
        
        # MLX path
        embedding = self._model.text_encoder(text)
        if isinstance(embedding, mx.array):
            embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten().astype(np.float32)
    
    def _encode_text_fallback(self, text: str) -> np.ndarray:
        """Encode text using open_clip fallback."""
        import torch
        
        tokens = self._tokenizer([text]).to(self._device)
        
        with torch.no_grad():
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().numpy().astype(np.float32)
    
    def unload(self):
        """Unload model and free memory."""
        print(f"Unloading CLIP model: {self.model_name}")
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False
        
        # Force garbage collection
        gc.collect()
        
        # Clear MLX cache if possible
        try:
            mx.metal.clear_cache()
        except:
            pass


# Current loaded model
_current_model: Optional[MLXClip] = None
_current_model_name: Optional[str] = None


def get_clip_model(model_name: str = "ViT-B-32__openai") -> MLXClip:
    """
    Get CLIP model, loading or switching as needed.
    
    If a different model is requested than currently loaded,
    unloads the current model first to free memory.
    """
    global _current_model, _current_model_name
    
    # Normalize model name
    normalized_name = model_name.replace("::", "__")
    
    # Check if we need to load a different model
    if _current_model is not None and _current_model_name != normalized_name:
        print(f"Switching CLIP model: {_current_model_name} -> {normalized_name}")
        _current_model.unload()
        _current_model = None
        _current_model_name = None
    
    # Load model if needed
    if _current_model is None:
        _current_model = MLXClip(normalized_name)
        _current_model_name = normalized_name
    
    return _current_model


# For testing
if __name__ == "__main__":
    import sys
    
    print("Testing MLX CLIP model...")
    
    # Test default model
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
    
    # Test model switching
    print("\nTesting model switch...")
    clip2 = get_clip_model("ViT-B-16__openai")
    text_emb2 = clip2.encode_text("a photo of a dog")
    print(f"New model embedding shape: {text_emb2.shape}")
    
    print("\n✅ CLIP test passed!")