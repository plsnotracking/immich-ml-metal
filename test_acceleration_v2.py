# test_acceleration.py
import time
import io
from PIL import Image

# Create a test image
img = Image.new("RGB", (640, 480), color=(100, 150, 200))
buf = io.BytesIO()
img.save(buf, format="JPEG")
test_bytes = buf.getvalue()

# Test face detection (uses Vision framework → ANE)
from src.models.face_detect import detect_faces

start = time.time()
for _ in range(20):
    detect_faces(test_bytes)
elapsed = time.time() - start
print(f"Face detection: {elapsed:.2f}s for 20 iterations ({elapsed/20*1000:.1f}ms each)")

# Test CLIP (uses MLX → Metal GPU)
from src.models.clip import get_clip_model

clip = get_clip_model()
start = time.time()
for _ in range(20):
    clip.encode_image(test_bytes)
elapsed = time.time() - start
print(f"CLIP encode: {elapsed:.2f}s for 20 iterations ({elapsed/20*1000:.1f}ms each)")