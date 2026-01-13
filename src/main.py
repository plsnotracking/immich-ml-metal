"""
immich-ml-metal: Metal/ANE-optimized ML service for Immich.

Drop-in replacement for Immich's ML service, optimized for Apple Silicon.
"""

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import ORJSONResponse, PlainTextResponse
from typing import Optional
import json
import numpy as np
from PIL import Image
import io
import os

from .config import settings

# Use real models unless STUB_MODE is set
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

app = FastAPI(
    title="immich-ml-metal",
    description="Metal/ANE-optimized drop-in replacement for Immich ML",
    version="0.1.0"
)

def get_clip(model_name: str = "ViT-B-32__openai"):
    """Get CLIP model, loading on first use or switching if model changed."""
    if STUB_MODE:
        return None
    from .models.clip import get_clip_model
    return get_clip_model(model_name)


def run_face_recognition(image_bytes: bytes, min_score: float, model_name: str) -> list[dict]:
    """Run face detection and embedding generation."""
    from .models.face_detect import detect_faces
    from .models.face_embed import get_face_embedding, get_face_embedding_from_bbox
    
    # Detect faces using Vision framework
    faces, _, _ = detect_faces(image_bytes)
    
    # Filter by score and generate embeddings
    results = []
    for face in faces:
        if face["score"] < min_score:
            continue
        
        # Generate embedding
        if "landmarks" in face:
            embedding = get_face_embedding(
                image_bytes,
                face["landmarks"],
                model_name
            )
        else:
            embedding = get_face_embedding_from_bbox(
                image_bytes,
                face["boundingBox"],
                model_name
            )
        
        if embedding is not None:
            # IMPORTANT: Immich expects embedding as a string, not an array
            results.append({
                "boundingBox": face["boundingBox"],
                "embedding": str(embedding.tolist()),
                "score": face["score"]
            })
    
    return results


@app.get("/")
async def root():
    """Root endpoint - mirrors Immich ML."""
    return ORJSONResponse({"message": "Immich ML"})


@app.get("/ping")
def ping():
    """Health check endpoint."""
    return PlainTextResponse("pong")


@app.post("/predict")
async def predict(
    entries: str = Form(...),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Main prediction endpoint - mirrors Immich ML API.
    
    Args:
        entries: JSON string describing requested tasks
        image: Optional image file for visual tasks
        text: Optional text for text encoding tasks
    
    Returns:
        ORJSONResponse with inference results
    """
    # Parse the entries JSON
    try:
        tasks = json.loads(entries)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid entries JSON: {e}")
    
    # Validate input
    if image is None and text is None:
        raise HTTPException(status_code=400, detail="Either image or text must be provided")
    
    response = {}
    
    # Read image if provided
    image_bytes = None
    img = None
    if image:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        response["imageHeight"] = img.height
        response["imageWidth"] = img.width
    
    # Process each requested task
    for task_type, task_config in tasks.items():
        
        if task_type == "clip":
            # CLIP embedding task
            if "visual" in task_config and image_bytes:
                model_name = task_config["visual"].get("modelName", settings.clip_model)
                
                if STUB_MODE:
                    # Stub: return random normalized embedding
                    embedding = np.random.randn(512).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)
                else:
                    # Real inference - pass model name for dynamic loading
                    clip = get_clip(model_name)
                    embedding = clip.encode_image(image_bytes)
                
                # IMPORTANT: Immich expects CLIP embedding as a string
                response["clip"] = str(embedding.tolist())
                
            elif "textual" in task_config and text:
                model_name = task_config["textual"].get("modelName", settings.clip_model)
                
                if STUB_MODE:
                    embedding = np.random.randn(512).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)
                else:
                    # Real inference - pass model name for dynamic loading
                    clip = get_clip(model_name)
                    embedding = clip.encode_text(text)
                
                # IMPORTANT: Immich expects CLIP embedding as a string
                response["clip"] = str(embedding.tolist())
        
        elif task_type == "facial-recognition":
            if image_bytes is None:
                continue
                
            detection_config = task_config.get("detection", {})
            recognition_config = task_config.get("recognition", {})
            
            min_score = detection_config.get("options", {}).get("minScore", settings.face_min_score)
            model_name = recognition_config.get("modelName", settings.face_model)
            
            if STUB_MODE:
                # Stub: return one fake face (with string embedding)
                fake_embedding = np.random.randn(512).astype(np.float32).tolist()
                faces = [
                    {
                        "boundingBox": {
                            "x1": int(img.width * 0.25),
                            "y1": int(img.height * 0.15),
                            "x2": int(img.width * 0.75),
                            "y2": int(img.height * 0.85)
                        },
                        "embedding": str(fake_embedding),
                        "score": 0.99
                    }
                ]
            else:
                # Real inference
                faces = run_face_recognition(image_bytes, min_score, model_name)
            
            response["facial-recognition"] = faces
        
        elif task_type == "ocr":
            if image_bytes is None:
                continue
                
            detection_config = task_config.get("detection", {})
            recognition_config = task_config.get("recognition", {})
            
            min_detection_score = detection_config.get("options", {}).get("minScore", 0.0)
            min_recognition_score = recognition_config.get("options", {}).get("minScore", 0.0)
            # Use the higher of the two thresholds
            min_score = max(min_detection_score, min_recognition_score)
            
            if STUB_MODE:
                # Stub: return fake OCR result
                response["ocr"] = {
                    "text": ["placeholder", "text"],
                    "box": [0, 0, 100, 50, 0, 50, 100, 100],
                    "boxScore": [0.95, 0.92],
                    "textScore": [0.98, 0.96]
                }
            else:
                # Real inference using Vision framework
                from .models.ocr import recognize_text
                response["ocr"] = recognize_text(image_bytes, min_confidence=min_score)
    
    return ORJSONResponse(response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )