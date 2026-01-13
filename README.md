# immich-ml-metal

A Metal/ANE-optimized drop-in replacement for [Immich's](https://immich.app/) machine learning service, designed specifically for Apple Silicon Macs.

## Features

- **CLIP Embeddings**: MLX-accelerated image and text embeddings for smart search
- **Face Detection**: Apple Vision framework for hardware-accelerated face detection
- **Face Recognition**: CoreML-accelerated ArcFace embeddings (InsightFace buffalo_l)
- **OCR**: Apple Vision framework text recognition

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- Immich server (to connect to)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/immich-ml-metal.git
cd immich-ml-metal

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e ".[dev]"
```

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the service
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 3003 --reload
```

The service will be available at `http://localhost:3003`.

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_HOST` | `0.0.0.0` | Bind address |
| `ML_PORT` | `3003` | Port number |
| `ML_MODELS_DIR` | `./models` | Model storage directory |
| `ML_CLIP_MODEL` | `ViT-B-32__openai` | CLIP model name |
| `ML_FACE_MODEL` | `buffalo_l` | Face recognition model |
| `ML_USE_COREML` | `true` | Enable CoreML acceleration |

## Connecting to Immich

Update your Immich configuration to point to this service:

```yaml
# docker-compose.yml or .env
MACHINE_LEARNING_URL=http://<mac-ip>:3003
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Test specific module
pytest tests/test_api.py -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check, returns "pong" |
| `/` | GET | Root, returns service info |
| `/predict` | POST | Main inference endpoint |

See [API Contract](docs/api-contract.md) for detailed request/response formats.

## Project Status

- [x] Phase 0: API contract documentation
- [x] Phase 1: Project setup
- [x] Phase 2: Stub service (placeholder responses)
- [ ] Phase 3: CLIP implementation (MLX)
- [ ] Phase 4: Face detection (Vision framework)
- [ ] Phase 5: Face embeddings (InsightFace + CoreML)
- [ ] Phase 6: OCR (Vision framework)
- [ ] Phase 7: Integration testing

## Acknowledgments

- [Immich](https://immich.app/) - The amazing self-hosted photo/video solution
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition library

## License

MIT