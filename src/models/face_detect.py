"""
Face detection using Apple's Vision framework.

Runs on the Neural Engine (ANE) for hardware acceleration.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, TypedDict

import objc
from Foundation import NSAutoreleasePool, NSData
from PIL import Image

import Vision

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class BoundingBox(TypedDict):
    """Bounding box coordinates in pixels."""
    x1: int
    y1: int
    x2: int
    y2: int


class FaceDetection(TypedDict, total=False):
    """Face detection result."""
    boundingBox: BoundingBox
    score: float
    landmarks: list[list[float]]


def detect_faces(image_bytes: bytes) -> tuple[list[FaceDetection], int, int]:
    """
    Detect faces using Apple's Vision framework.
    
    Args:
        image_bytes: Raw image data (JPEG, PNG, etc.)
        
    Returns:
        Tuple of (faces, image_width, image_height).
        Each face dict contains:
          - boundingBox: {x1, y1, x2, y2} in pixels
          - score: confidence score
          - landmarks: 5-point landmarks for alignment (if available)
          
    Raises:
        ValueError: If image data is invalid.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_image.size
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise ValueError(f"Invalid image data: {e}") from e
    
    # Use autorelease pool to prevent memory accumulation in long-running service
    pool = NSAutoreleasePool.alloc().init()
    try:
        return _detect_faces_impl(image_bytes, img_width, img_height)
    finally:
        del pool


def _detect_faces_impl(
    image_bytes: bytes, 
    img_width: int, 
    img_height: int
) -> tuple[list[FaceDetection], int, int]:
    """Internal face detection implementation (assumes autorelease pool is active)."""
    try:
        ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        
        success, error = handler.performRequests_error_([request], None)
        
        if not success or error:
            logger.error(f"Vision framework error: {error}")
            return [], img_width, img_height
        
        faces: list[FaceDetection] = []
        results = request.results() or []
        
        for observation in results:
            bbox = observation.boundingBox()
            
            # Convert to pixel coordinates (flip Y axis - Vision uses bottom-left origin)
            x1 = bbox.origin.x * img_width
            y1 = (1.0 - bbox.origin.y - bbox.size.height) * img_height
            x2 = (bbox.origin.x + bbox.size.width) * img_width
            y2 = (1.0 - bbox.origin.y) * img_height
            
            face_data: FaceDetection = {
                "boundingBox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
                "score": float(observation.confidence()),
            }
            
            landmarks = observation.landmarks()
            if landmarks:
                five_points = _extract_five_point_landmarks(landmarks, img_width, img_height)
                if five_points is not None:
                    face_data["landmarks"] = five_points
            
            faces.append(face_data)
        
        logger.debug(f"Detected {len(faces)} face(s) in {img_width}x{img_height} image")
        return faces, img_width, img_height
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}", exc_info=True)
        return [], img_width, img_height


def _extract_five_point_landmarks(
    landmarks: Vision.VNFaceLandmarks2D,
    img_width: int,
    img_height: int,
) -> list[list[float]] | None:
    """
    Extract 5 landmark points for ArcFace alignment:
    - Left eye center
    - Right eye center  
    - Nose tip
    - Left mouth corner
    - Right mouth corner
    
    Returns list of [x, y] points in pixel coordinates, or None if not available.
    """
    def get_region_points(region: object) -> list:
        """Convert PyObjC varlist to Python list of points."""
        if region is None:
            return []
        point_count = region.pointCount()
        if point_count == 0:
            return []
        raw_points = region.normalizedPoints()
        return [raw_points[i] for i in range(point_count)]
    
    def get_region_center(region: object) -> list[float] | None:
        """Get center point of a landmark region."""
        points = get_region_points(region)
        if not points:
            return None
        
        x_sum = sum(p.x for p in points)
        y_sum = sum(p.y for p in points)
        n = len(points)
        
        # Convert to pixels (flip Y)
        x = (x_sum / n) * img_width
        y = (1.0 - y_sum / n) * img_height
        return [x, y]
    
    try:
        # Left eye center
        left_eye = get_region_center(landmarks.leftEye())
        
        # Right eye center
        right_eye = get_region_center(landmarks.rightEye())
        
        # Nose tip - use last point of nose region
        nose: list[float] | None = None
        nose_points = get_region_points(landmarks.nose())
        if nose_points:
            p = nose_points[-1]
            nose = [p.x * img_width, (1.0 - p.y) * img_height]
        
        # Mouth corners - find leftmost and rightmost points by x-coordinate
        # (Vision framework doesn't guarantee point ordering in contours)
        left_mouth: list[float] | None = None
        right_mouth: list[float] | None = None
        outer_lips_points = get_region_points(landmarks.outerLips())
        if outer_lips_points:
            # Convert all points to pixel coordinates
            lips_px = [
                (p.x * img_width, (1.0 - p.y) * img_height) 
                for p in outer_lips_points
            ]
            # Find extremes by x-coordinate
            left_mouth = list(min(lips_px, key=lambda p: p[0]))
            right_mouth = list(max(lips_px, key=lambda p: p[0]))
        
        # All 5 points must be present
        if all([left_eye, right_eye, nose, left_mouth, right_mouth]):
            return [left_eye, right_eye, nose, left_mouth, right_mouth]  # type: ignore[list-item]
        
        logger.debug("Could not extract all 5 landmark points")
        return None
        
    except Exception as e:
        logger.warning(f"Landmark extraction failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    
    match sys.argv[1:]:
        case []:
            logger.info("Usage: python -m src.models.face_detect <image_path>")
            logger.info("Creating test with a blank image...")
            
            test_img = Image.new("RGB", (640, 480), color=(200, 180, 170))
            buffer = io.BytesIO()
            test_img.save(buffer, format="JPEG")
            test_bytes = buffer.getvalue()
        case [image_path, *_]:
            test_bytes = Path(image_path).read_bytes()
    
    logger.info("Testing Vision framework face detection...")
    faces, width, height = detect_faces(test_bytes)
    
    logger.info(f"Image size: {width}x{height}")
    logger.info(f"Faces detected: {len(faces)}")
    
    for i, face in enumerate(faces):
        logger.info(f"\nFace {i + 1}:")
        logger.info(f"  Bounding box: {face['boundingBox']}")
        logger.info(f"  Score: {face['score']:.3f}")
        match "landmarks" in face:
            case True:
                logger.info("  Landmarks (5-point): ✓")
            case False:
                logger.info("  Landmarks: not available")
    
    logger.info("\n✅ Face detection test complete!")
