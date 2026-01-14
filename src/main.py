"""
immich-ml-metal: Metal/ANE-optimized ML service for Immich.

Drop-in replacement for Immich's ML service, optimized for Apple Silicon.
"""

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse, JSONResponse
from typing import Optional
import json
import numpy as np
from PIL import Image
import io
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use real models unless STUB_MODE is set
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

if STUB_MODE:
    logger.warning("Running in STUB_MODE - returning fake data")


# Pydantic models for response validation
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class FaceDetection(BaseModel):
    boundingBox: BoundingBox
    embedding: str  # Stringified array
    score: float


class OCRResult(BaseModel):
    text: list[str]
    box: list[int]  # Flat list of coordinates
    boxScore: list[float]
    textScore: list[float]


class PredictResponse(BaseModel):
    imageHeight: Optional[int] = None
    imageWidth: Optional[int] = None
    clip: Optional[str] = None  # Stringified array
    facial_recognition: Optional[list[FaceDetection]] = Field(None, alias="facial-recognition")
    ocr: Optional[OCRResult] = None


# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    logger.info("Starting immich-ml-metal service")
    logger.info(f"STUB_MODE: {STUB_MODE}")
    logger.info(f"Settings: host={settings.host}, port={settings.port}")
    logger.info(f"CLIP model: {settings.clip_model}")
    logger.info(f"Face model: {settings.face_model}")
    logger.info(f"Face min score: {settings.face_min_score}")
    logger.info(f"Max image size: {settings.max_image_size / 1024 / 1024:.1f}MB")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down immich-ml-metal service")
    if not STUB_MODE:
        try:
            from .models.clip import _current_model, _model_lock
            from .models.face_embed import unload_recognition_model
            
            with _model_lock:
                if _current_model:
                    _current_model.unload()
            
            unload_recognition_model()
            logger.info("Models unloaded successfully")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")


app = FastAPI(
    title="immich-ml-metal",
    description="Metal/ANE-optimized drop-in replacement for Immich ML",
    version="0.1.0",
    lifespan=lifespan
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response


def get_clip(model_name: str = "ViT-B-32__openai"):
    """Get CLIP model, loading on first use or switching if model changed."""
    if STUB_MODE:
        return None
    from .models.clip import get_clip_model
    return get_clip_model(model_name)


async def run_face_recognition_async(
    image_bytes: bytes, 
    min_score: float, 
    model_name: str
) -> list[dict]:
    """Run face detection and embedding generation (async wrapper)."""
    # Run blocking operations in thread pool
    return await asyncio.to_thread(
        _run_face_recognition_sync,
        image_bytes,
        min_score,
        model_name
    )


def _run_face_recognition_sync(
    image_bytes: bytes,
    min_score: float,
    model_name: str
) -> list[dict]:
    """Synchronous face recognition implementation."""
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
            embedding = get_face_embedding(image_bytes, face["landmarks"], model_name)
        else:
            embedding = get_face_embedding_from_bbox(image_bytes, face["boundingBox"], model_name)
        
        if embedding is not None:
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


@app.get("/health")
async def health():
    """
    Detailed health check endpoint.
    
    Checks if models can be loaded and basic functionality works.
    """
    health_status = {
        "status": "healthy",
        "stub_mode": STUB_MODE,
        "checks": {}
    }
    
    try:
        if not STUB_MODE:
            # Check CLIP model
            try:
                clip = get_clip(settings.clip_model)
                health_status["checks"]["clip"] = "ok"
            except Exception as e:
                logger.error(f"CLIP health check failed: {e}")
                health_status["checks"]["clip"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check face model
            try:
                from .models.face_embed import get_recognition_model
                get_recognition_model(settings.face_model)
                health_status["checks"]["face_recognition"] = "ok"
            except Exception as e:
                logger.error(f"Face recognition health check failed: {e}")
                health_status["checks"]["face_recognition"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # Vision framework checks (face detection and OCR) are lightweight
            health_status["checks"]["vision_framework"] = "ok"
        else:
            health_status["checks"]["stub_mode"] = "active"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e)
            },
            status_code=503
        )


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
        logger.error(f"Invalid entries JSON: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid entries JSON: {e}")
    
    # Validate input
    if image is None and text is None:
        raise HTTPException(
            status_code=400, 
            detail="Either image or text must be provided"
        )
    
    response = {}
    
    # Read and validate image if provided
    image_bytes = None
    img = None
    if image:
        # Check file size
        if image.size and image.size > settings.max_image_size:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Max size: {settings.max_image_size / 1024 / 1024:.1f}MB"
            )
        
        try:
            image_bytes = await image.read()
            img = Image.open(io.BytesIO(image_bytes))
            response["imageHeight"] = img.height
            response["imageWidth"] = img.width
            
            # Validate image isn't ridiculously large
            if img.width * img.height > 100_000_000:  # 100 megapixels
                raise HTTPException(
                    status_code=413,
                    detail="Image resolution too high"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to read/decode image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
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
                    # Real inference - run in thread pool to avoid blocking
                    clip = get_clip(model_name)
                    embedding = await asyncio.to_thread(
                        clip.encode_image,
                        image_bytes
                    )
                
                response["clip"] = str(embedding.tolist())
                
            elif "textual" in task_config and text:
                model_name = task_config["textual"].get("modelName", settings.clip_model)
                
                if STUB_MODE:
                    embedding = np.random.randn(512).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)
                else:
                    # Real inference
                    clip = get_clip(model_name)
                    embedding = await asyncio.to_thread(
                        clip.encode_text,
                        text
                    )
                
                response["clip"] = str(embedding.tolist())
        
        elif task_type == "facial-recognition":
            if image_bytes is None:
                continue
                
            detection_config = task_config.get("detection", {})
            recognition_config = task_config.get("recognition", {})
            
            # Use Immich's min_score if provided, otherwise use our default
            min_score = detection_config.get("options", {}).get(
                "minScore", 
                settings.face_min_score
            )
            model_name = recognition_config.get("modelName", settings.face_model)
            
            if STUB_MODE:
                # Stub: return one fake face
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
                faces = await run_face_recognition_async(
                    image_bytes,
                    min_score,
                    model_name
                )
            
            response["facial-recognition"] = faces
        
        elif task_type == "ocr":
            if image_bytes is None:
                continue
                
            detection_config = task_config.get("detection", {})
            recognition_config = task_config.get("recognition", {})
            
            min_detection_score = detection_config.get("options", {}).get("minScore", 0.0)
            min_recognition_score = recognition_config.get("options", {}).get("minScore", 0.0)
            min_score = max(min_detection_score, min_recognition_score)
            
            if STUB_MODE:
                # Stub: return fake OCR result
                response["ocr"] = {
                    "text": ["placeholder", "text"],
                    "box": [0, 0, 100, 0, 100, 50, 0, 50, 0, 50, 100, 50, 100, 100, 0, 100],
                    "boxScore": [0.95, 0.92],
                    "textScore": [0.98, 0.96]
                }
            else:
                # Real inference - run in thread pool
                from .models.ocr import recognize_text
                ocr_result = await asyncio.to_thread(
                    recognize_text,
                    image_bytes,
                    min_confidence=min_score
                )
                response["ocr"] = ocr_result
    
    # Validate response against schema
    try:
        validated_response = PredictResponse(**response)
        return ORJSONResponse(validated_response.model_dump(by_alias=True, exclude_none=True))
    except Exception as e:
        logger.error(f"Response validation failed: {e}", exc_info=True)
        # Return anyway but log the issue
        return ORJSONResponse(response)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.host == "0.0.0.0" else "Internal server error"
        }
    )


def main():
    """Entry point for running the service directly."""
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,  # Don't enable reload in production
        log_level="info",
        timeout_keep_alive=settings.request_timeout,
    )


if __name__ == "__main__":
    main()