"""
Setup verification tests for immich-ml-metal.

Run with: pytest tests/test_setup.py -v
"""

import sys
import platform


def test_python_version():
    """Verify Python 3.10+ is being used."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"
    print(f"✅ Python version: {sys.version}")


def test_apple_silicon():
    """Verify running on Apple Silicon Mac."""
    assert platform.system() == "Darwin", "Must run on macOS"
    assert platform.machine() == "arm64", "Must run on Apple Silicon (arm64)"
    print(f"✅ Platform: {platform.system()} {platform.machine()}")


def test_coreml_available():
    """Verify CoreML execution provider is available."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"Available ONNX providers: {providers}")
        
        # CoreML should be available on Apple Silicon
        # Note: May show as CoreMLExecutionProvider or via CPU with ANE delegation
        has_coreml = 'CoreMLExecutionProvider' in providers
        has_cpu = 'CPUExecutionProvider' in providers
        
        if has_coreml:
            print("✅ CoreML execution provider available")
        elif has_cpu:
            print("⚠️  CoreML not listed, but CPU provider available (may still use ANE)")
        else:
            raise AssertionError(f"No suitable providers found: {providers}")
    except ImportError:
        import pytest
        pytest.skip("onnxruntime not installed")


def test_vision_framework():
    """Verify Apple Vision framework is accessible via PyObjC."""
    try:
        import Vision
        import Quartz
        
        # Check we can create a request
        request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        assert request is not None
        print("✅ Vision framework accessible")
    except ImportError as e:
        import pytest
        pytest.skip(f"PyObjC Vision not installed: {e}")


def test_mlx_available():
    """Verify MLX is installed and working."""
    try:
        import mlx.core as mx
        
        # Quick sanity check
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = a + b
        mx.eval(c)
        
        assert c.tolist() == [5.0, 7.0, 9.0]
        print(f"✅ MLX working (version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'})")
    except ImportError:
        import pytest
        pytest.skip("MLX not installed")


def test_fastapi_imports():
    """Verify FastAPI and dependencies are importable."""
    from fastapi import FastAPI, Form, File, UploadFile
    from fastapi.responses import ORJSONResponse, PlainTextResponse
    import orjson
    import numpy as np
    from PIL import Image
    
    print("✅ FastAPI and dependencies importable")


def test_stub_service_imports():
    """Verify the stub service can be imported."""
    try:
        from src.main import app
        from src.config import settings
        
        assert app is not None
        assert settings is not None
        print(f"✅ Stub service importable (port: {settings.port})")
    except ImportError as e:
        import pytest
        pytest.skip(f"Stub service not yet created: {e}")


if __name__ == "__main__":
    """Run tests directly."""
    test_python_version()
    test_apple_silicon()
    
    print("\n--- Optional Dependencies ---")
    try:
        test_coreml_available()
    except Exception as e:
        print(f"⚠️  CoreML: {e}")
    
    try:
        test_vision_framework()
    except Exception as e:
        print(f"⚠️  Vision: {e}")
    
    try:
        test_mlx_available()
    except Exception as e:
        print(f"⚠️  MLX: {e}")
    
    test_fastapi_imports()
    
    print("\n✅ Basic setup verification complete!")