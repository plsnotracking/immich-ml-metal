import time
import io
import sys
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
import onnxruntime as ort

# Create test image
img = Image.new('RGB', (112, 112), color=(100, 150, 200))
buf = io.BytesIO()
img.save(buf, format='JPEG')

# Create a fake aligned face (112x112 is what ArcFace expects)
import cv2
face_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

# Test with CoreML (hardware accelerated)
print('Loading model with CoreML...')
from insightface.model_zoo import get_model
model_coreml = get_model('buffalo_s', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
model_coreml.prepare(ctx_id=0)

start = time.time()
for _ in range(20):
    model_coreml.get_feat(face_img)
elapsed = time.time() - start
print(f'CoreML: {elapsed:.2f}s for 20 iterations ({elapsed/20*1000:.1f}ms each)')

# Test with CPU only
print('\nLoading model with CPU only...')
model_cpu = get_model('buffalo_s', providers=['CPUExecutionProvider'])
model_cpu.prepare(ctx_id=0)

start = time.time()
for _ in range(20):
    model_cpu.get_feat(face_img)
elapsed = time.time() - start
print(f'CPU only: {elapsed:.2f}s for 20 iterations ({elapsed/20*1000:.1f}ms each)')