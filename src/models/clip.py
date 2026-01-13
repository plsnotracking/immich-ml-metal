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

# Model name mapping: Immich name -> MLX repo (or None to use open_clip fallback)
# Immich uses format like "ViT-B-32__openai" or "ViT-B-32::openai"
MODEL_MAP = {
    # OpenAI CLIP models -> MLX
    "ViT-B-32__openai": "mlx-community/clip-vit-base-patch32",
    "ViT-B-16__openai": "mlx-community/clip-vit-base-patch16",
    "ViT-L-14__openai": "mlx-community/clip-vit-large-patch14",
    
    # LAION CLIP models -> MLX (if available) or fallback
    "ViT-B-32__laion2b-s34b-b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    "ViT-B-32__laion2b_s34b_b79k": "mlx-community/clip-vit-base-patch32-laion2b",
    
    # SigLIP models -> None (use open_clip fallback)
    # These don't have MLX versions yet
    "ViT-B-16-SigLIP__webli": None,
    "ViT-B-16-SigLIP2__webli": None,
    "ViT-SO400M-16-SigLIP2-384__webli": None,
    
    # Default fallback
    "default": "mlx-community/clip-vit-base-patch32",
}

# open_clip model name mappings for fallback
# Maps Immich names to (arch, pretrained) tuples for open_clip
OPENCLIP_MAP = {
    # Standard CLIP
    "ViT-B-32__openai": ("ViT-B-32-quickgelu", "openai"),
    "ViT-B-16__openai": ("ViT-B-16", "openai"),
    "ViT-L-14__openai": ("ViT-L-14", "openai"),
    
    # LAION
    "ViT-B-32__laion2b-s34b-b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B-32__laion2b_s34b_b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    
    # SigLIP (supported by open_clip)
    "ViT-B-16-SigLIP__webli": ("ViT-B-16-SigLIP", "webli"),
    "ViT-B-16-SigLIP2__webli": ("ViT-B-16-SigLIP2", "webli"),
    "ViT-SO400M-16-SigLIP2-384__webli": ("ViT-SO400M-16-SigLIP2-384", "webli"),
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
        """Load the MLX CLIP model, or fallback to open_clip."""
        # Check if we have an MLX version
        self._repo_id = MODEL_MAP.get(self.model_name)
        
        # If unknown model and not a known open_clip model, use MLX default
        if self._repo_id is None and self.model_name not in OPENCLIP_MAP:
            print(f"⚠️ Unknown model '{self.model_name}', using MLX default (ViT-B-32)")
            self._repo_id = MODEL_MAP["default"]
        
        if self._repo_id is None:
            # Known open_clip model (e.g., SigLIP) - use open_clip fallback
            print(f"⚠️ No MLX version for {self.model_name}, using open_clip fallback")
            self._load_fallback()
            return
        
        try:
            from mlx_clip import mlx_clip
            
            print(f"Loading MLX CLIP model: {self.model_name} -> {self._repo_id}")
            
            # mlx_clip handles downloading and loading
            self._model = mlx_clip(self._repo_id)
            self._loaded = True
            
            print(f"✅ Loaded CLIP model via MLX: {self.model_name}")
            
        except ImportError:
            # Fallback to open_clip with MPS if mlx_clip not available
            print("⚠️ mlx_clip not available, falling back to open_clip with MPS")
            self._load_fallback()
        except Exception as e:
            print(f"⚠️ MLX load failed ({e}), falling back to open_clip")
            self._load_fallback()
    
    def _load_fallback(self):
        """Fallback to open_clip with MPS acceleration."""
        import torch
        import open_clip
        
        # Check if we have a known mapping
        if self.model_name in OPENCLIP_MAP:
            arch, pretrained = OPENCLIP_MAP[self.model_name]
        elif "__" in self.model_name:
            arch, pretrained = self.model_name.split("__", 1)
            # OpenAI models need quickgelu variant
            if pretrained == "openai" and "quickgelu" not in arch.lower() and "siglip" not in arch.lower():
                arch = arch + "-quickgelu"
        else:
            arch = "ViT-B-32-quickgelu"
            pretrained = "openai"
        
        print(f"Loading open_clip model: {arch} / {pretrained}")
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(arch)
        except Exception as e:
            print(f"⚠️ Failed to load {arch}/{pretrained}: {e}")
            print("Falling back to ViT-B-32-quickgelu/openai")
            arch, pretrained = "ViT-B-32-quickgelu", "openai"
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
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
        
        print(f"✅ Loaded CLIP model via open_clip: {arch}/{pretrained}")
    
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
    
    print("Testing CLIP model loading...")
    print(f"\nSupported MLX models: {[k for k, v in MODEL_MAP.items() if v is not None and k != 'default']}")
    print(f"Supported open_clip models: {list(OPENCLIP_MAP.keys())}")
    
    # Test default model (MLX)
    print("\n--- Testing MLX model ---")
    clip = get_clip_model("ViT-B-32__openai")
    text_emb = clip.encode_text("a photo of a cat")
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Text embedding norm: {np.linalg.norm(text_emb):.4f}")
    
    # Test image encoding if file provided
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            img_emb = clip.encode_image(f.read())
        print(f"Image embedding shape: {img_emb.shape}")
        similarity = np.dot(text_emb, img_emb)
        print(f"Text-image similarity: {similarity:.4f}")
    
    # Test SigLIP model (open_clip fallback)
    print("\n--- Testing SigLIP model (open_clip fallback) ---")
    clip2 = get_clip_model("ViT-B-16-SigLIP__webli")
    text_emb2 = clip2.encode_text("a photo of a dog")
    print(f"SigLIP embedding shape: {text_emb2.shape}")
    print(f"SigLIP embedding norm: {np.linalg.norm(text_emb2):.4f}")
    
    print("\n✅ CLIP tests passed!")