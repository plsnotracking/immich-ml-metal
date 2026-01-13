import sys
sys.path.insert(0, '.')
from src.models.clip import get_clip_model

# Test default model
clip = get_clip_model('ViT-B-32__openai')
emb = clip.encode_text('hello world')
print(f'Embedding shape: {emb.shape}')
print(f'Using MLX: {not hasattr(clip, "_use_fallback") or not clip._use_fallback}')