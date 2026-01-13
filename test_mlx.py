import sys
sys.path.insert(0, '.')
from src.models.clip import get_clip_model

clip = get_clip_model('ViT-B-32__openai')
emb = clip.encode_text('hello world')
print(f'Embedding shape: {emb.shape}')

# Check which backend is being used
if hasattr(clip, '_use_fallback') and clip._use_fallback:
    print(f'Backend: open_clip with {clip._device}')
else:
    print('Backend: MLX')