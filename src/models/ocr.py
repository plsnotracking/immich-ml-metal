"""
OCR (Optical Character Recognition) using Apple's Vision framework.

Uses VNRecognizeTextRequest for hardware-accelerated text recognition on Apple Silicon.
"""

import numpy as np
from PIL import Image
import io
from typing import Optional
import logging
from Foundation import NSData
import Vision

logger = logging.getLogger(__name__)


def recognize_text(
    image_bytes: bytes,
    min_confidence: float = 0.0,
    recognition_level: str = "accurate"
) -> dict:
    """
    Perform OCR using Apple's Vision framework.
    
    Args:
        image_bytes: Raw image data (JPEG, PNG, etc.)
        min_confidence: Minimum confidence threshold (0.0 - 1.0)
        recognition_level: "accurate" or "fast"
        
    Returns:
        Dict matching Immich OCR response format:
        {
            "text": [str, ...],
            "box": [x1, y1, x2, y2, x3, y3, x4, y4, ...],  # 8 coords per text
            "boxScore": [float, ...],
            "textScore": [float, ...]
        }
    """
    try:
        # Load image to get dimensions
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_image.size
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return {"text": [], "box": [], "boxScore": [], "textScore": []}
    
    try:
        # Create NSData from image bytes
        ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        
        # Create Vision image request handler
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        
        # Create text recognition request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        
        # Set recognition level
        if recognition_level == "fast":
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
        else:
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        
        # Enable language correction for better accuracy
        request.setUsesLanguageCorrection_(True)
        
        # Perform the request
        success, error = handler.performRequests_error_([request], None)
        
        if not success or error:
            logger.error(f"Vision OCR error: {error}")
            return {"text": [], "box": [], "boxScore": [], "textScore": []}
        
        # Process results
        texts = []
        boxes = []
        box_scores = []
        text_scores = []
        
        results = request.results() or []
        
        for observation in results:
            # Get the top candidate
            candidates = observation.topCandidates_(1)
            if not candidates or len(candidates) == 0:
                continue
            
            candidate = candidates[0]
            confidence = candidate.confidence()
            
            # Filter by confidence
            if confidence < min_confidence:
                continue
            
            text = candidate.string()
            texts.append(text)
            text_scores.append(float(confidence))
            
            # Get bounding box (normalized coordinates, origin at bottom-left)
            bbox = observation.boundingBox()
            
            # Convert to pixel coordinates (flip Y axis)
            x = bbox.origin.x * img_width
            y = (1.0 - bbox.origin.y - bbox.size.height) * img_height
            w = bbox.size.width * img_width
            h = bbox.size.height * img_height
            
            # Immich expects 8 coordinates per box (quadrilateral corners)
            # Order: top-left, top-right, bottom-right, bottom-left (clockwise)
            x1, y1 = int(x), int(y)              # top-left
            x2, y2 = int(x + w), int(y)          # top-right
            x3, y3 = int(x + w), int(y + h)      # bottom-right
            x4, y4 = int(x), int(y + h)          # bottom-left
            
            # Add all 8 coordinates to the flat list
            boxes.extend([x1, y1, x2, y2, x3, y3, x4, y4])
            
            # Box score (use observation confidence if available)
            box_scores.append(
                float(observation.confidence()) 
                if hasattr(observation, 'confidence') 
                else float(confidence)
            )
        
        logger.debug(f"OCR detected {len(texts)} text region(s)")
        
        return {
            "text": texts,
            "box": boxes,
            "boxScore": box_scores,
            "textScore": text_scores
        }
        
    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return {"text": [], "box": [], "boxScore": [], "textScore": []}


if __name__ == "__main__":
    import sys
    
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.models.ocr <image_path>")
        logger.info("Creating test image with text...")
        
        from PIL import ImageDraw
        
        img = Image.new("RGB", (400, 200), color="white")
        draw = ImageDraw.Draw(img)
        
        # Draw some text
        draw.text((20, 30), "Hello World!", fill="black")
        draw.text((20, 80), "immich-ml-metal", fill="blue")
        draw.text((20, 130), "OCR Test 123", fill="darkgreen")
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        test_bytes = buffer.getvalue()
        
        # Also save to file for inspection
        img.save("ocr_test.png")
        logger.info("Saved test image to ocr_test.png")
    else:
        with open(sys.argv[1], "rb") as f:
            test_bytes = f.read()
    
    logger.info("Testing Vision framework OCR...")
    result = recognize_text(test_bytes)
    
    logger.info(f"Detected {len(result['text'])} text region(s):")
    for i, text in enumerate(result['text']):
        score = result['textScore'][i]
        # Get box coordinates (8 values per box)
        box_start = i * 8
        coords = result['box'][box_start:box_start + 8]
        logger.info(f"  [{score:.2f}] \"{text}\"")
        logger.info(f"         Box: {coords}")
    
    logger.info("\n✅ OCR test complete!")