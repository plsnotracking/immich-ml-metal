"""
API endpoint tests for immich-ml-metal.

Run with: pytest tests/test_api.py -v
"""

import pytest
import json
from io import BytesIO
from PIL import Image

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from src.main import app
    return TestClient(app)


@pytest.fixture
def test_image_bytes():
    """Create a simple test image."""
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_ping(self, client):
        """Test /ping returns pong."""
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.text == "pong"
    
    def test_root(self, client):
        """Test / returns expected message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Immich ML"


class TestCLIPEndpoints:
    """Test CLIP embedding endpoints."""
    
    def test_clip_visual(self, client, test_image_bytes):
        """Test CLIP visual embedding."""
        entries = {"clip": {"visual": {"modelName": "ViT-B-32__openai"}}}
        
        response = client.post(
            "/predict",
            data={"entries": json.dumps(entries)},
            files={"image": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "clip" in data
        assert "imageHeight" in data
        assert "imageWidth" in data
        
        # Check embedding dimensions (should be 512 for ViT-B-32)
        embedding = data["clip"]
        assert isinstance(embedding, list)
        assert len(embedding) == 512
        
        # Check image dimensions captured
        assert data["imageHeight"] == 480
        assert data["imageWidth"] == 640
    
    def test_clip_textual(self, client):
        """Test CLIP text embedding."""
        entries = {"clip": {"textual": {"modelName": "ViT-B-32__openai"}}}
        
        response = client.post(
            "/predict",
            data={
                "entries": json.dumps(entries),
                "text": "a photo of a cat"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "clip" in data
        embedding = data["clip"]
        assert isinstance(embedding, list)
        assert len(embedding) == 512


class TestFaceRecognition:
    """Test facial recognition endpoints."""
    
    def test_face_detection(self, client, test_image_bytes):
        """Test face detection and embedding."""
        entries = {
            "facial-recognition": {
                "detection": {
                    "modelName": "buffalo_l",
                    "options": {"minScore": 0.7}
                },
                "recognition": {
                    "modelName": "buffalo_l"
                }
            }
        }
        
        response = client.post(
            "/predict",
            data={"entries": json.dumps(entries)},
            files={"image": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "facial-recognition" in data
        assert "imageHeight" in data
        assert "imageWidth" in data
        
        faces = data["facial-recognition"]
        assert isinstance(faces, list)
        
        # Check face structure (if faces detected)
        if len(faces) > 0:
            face = faces[0]
            assert "boundingBox" in face
            assert "embedding" in face
            assert "score" in face
            
            bbox = face["boundingBox"]
            assert all(k in bbox for k in ["x1", "y1", "x2", "y2"])
            
            # Embedding should be 512-dim
            assert len(face["embedding"]) == 512


class TestOCR:
    """Test OCR endpoints."""
    
    def test_ocr(self, client, test_image_bytes):
        """Test OCR text recognition."""
        entries = {
            "ocr": {
                "detection": {
                    "modelName": "paddle-ocr",
                    "options": {"minScore": 0.5, "maxResolution": 1024}
                },
                "recognition": {
                    "modelName": "paddle-ocr",
                    "options": {"minScore": 0.5}
                }
            }
        }
        
        response = client.post(
            "/predict",
            data={"entries": json.dumps(entries)},
            files={"image": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ocr" in data
        ocr_result = data["ocr"]
        
        # Check OCR response structure
        assert "text" in ocr_result
        assert "box" in ocr_result
        assert "boxScore" in ocr_result
        assert "textScore" in ocr_result


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_json(self, client, test_image_bytes):
        """Test handling of invalid JSON in entries."""
        response = client.post(
            "/predict",
            data={"entries": "not valid json"},
            files={"image": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 422
    
    def test_missing_input(self, client):
        """Test handling of missing image and text."""
        entries = {"clip": {"visual": {"modelName": "test"}}}
        
        response = client.post(
            "/predict",
            data={"entries": json.dumps(entries)}
        )
        
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])