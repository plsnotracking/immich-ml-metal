# immich-ml-metal

> **⚠️ Important Disclaimer**: I (the repository owner) am not a software developer. This project was architected, designed, and primarily authored by Claude (Anthropic's AI assistant) based on my requirements and feedback. While I've tested it in my home environment, please treat this as an **experimental community project** rather than production-ready software.
>
> **For Immich developers reviewing this**: I'm sharing this in hopes it's useful for the Mac community, but I fully acknowledge this needs proper vetting. Constructive feedback is very welcome!

A Metal/ANE-optimized drop-in replacement for [Immich's](https://immich.app/) machine learning service, designed specifically for Apple Silicon Macs. This allows Mac users to run Immich's ML workloads natively on their hardware instead of requiring Docker Desktop.

## What This Does

Immich's standard ML container runs CUDA-optimized models intended for NVIDIA GPUs. This project reimplements the same ML API using Apple's frameworks:

- **CLIP Embeddings**: MLX-accelerated (with open_clip fallback) for image/text search
- **Face Detection**: Apple Vision framework (runs on Neural Engine)
- **Face Recognition**: InsightFace ArcFace with CoreML acceleration
- **OCR**: Apple Vision framework text recognition

## Project Status

**⚠️ Alpha Quality - Use at Your Own Risk**

- [x] Phase 0: API contract documentation
- [x] Phase 1: Project setup  
- [x] Phase 2: Stub service (placeholder responses)
- [x] Phase 3: CLIP implementation (MLX + open_clip fallback)
- [x] Phase 4: Face detection (Vision framework)
- [x] Phase 5: Face embeddings (InsightFace + CoreML)
- [x] Phase 6: OCR (Vision framework)
- [ ] Phase 7: Comprehensive integration testing with real Immich instance
- [ ] Phase 8: Community testing and validation
- [ ] Phase 9: Docker support (if desired)

**Known Limitations:**
- Only tested in my specific home setup (Mac Mini M4, macOS 15.2, Immich v1.122.x)
- Not all Immich ML features may be fully compatible
- Memory usage not extensively optimized
- No load testing performed
- Configuration may need tuning for your specific use case

## Requirements

- **macOS 13+** (Ventura or later)
- **Apple Silicon Mac** (M1/M2/M3/M4 - Intel Macs not supported)
- **Python 3.10-3.12**
- **Immich server** already running (this replaces just the ML service)
- **~2GB disk space** for models (first-run download)
- **Sufficient RAM** - recommend 16GB+ for comfortable operation

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

# First run will download models (~500MB-2GB depending on choices)
# This may take 5-15 minutes
python -m src.main
```

**⚠️ First Run**: Models will auto-download from HuggingFace and InsightFace on first use. Be patient and watch the console output.

## Configuration

Configure via environment variables or edit `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_HOST` | `0.0.0.0` | Bind address |
| `ML_PORT` | `3003` | Port number (must match Immich config) |
| `ML_MODELS_DIR` | `./models` | Model storage directory |
| `ML_CLIP_MODEL` | `ViT-B-32__openai` | CLIP model name |
| `ML_FACE_MODEL` | `buffalo_l` | Face recognition model (buffalo_s/m/l) |
| `ML_FACE_MIN_SCORE` | `0.7` | Face detection confidence threshold |
| `ML_USE_COREML` | `true` | Enable CoreML acceleration |
| `ML_USE_ANE` | `true` | Enable Apple Neural Engine |

**⚠️ Configuration Gotcha**: There's a known inconsistency in `config.py` between default values and environment variable defaults. The face_min_score particularly varies between 0.034 and 0.7. If you experience issues with face detection, try adjusting `ML_FACE_MIN_SCORE`.

### Model Choices

**CLIP Models** (for smart search):
- `ViT-B-32__openai` - Smaller, faster (~350MB) - **recommended for most users**
- `ViT-B-16__openai` - Balanced (~350MB)
- `ViT-L-14__openai` - Larger, more accurate (~1.7GB) - requires more RAM
- `ViT-B-16-SigLIP__webli` - Alternative architecture (uses open_clip fallback)

**Face Models**:
- `buffalo_s` - Smallest, fastest (~60MB total pack)
- `buffalo_m` - Balanced (~150MB total pack)
- `buffalo_l` - Most accurate (~350MB total pack) - **default**

## Connecting to Immich

In your Immich `docker-compose.yml` or `.env`:

```yaml
# Point to your Mac's IP address (not localhost unless Immich is also native)
MACHINE_LEARNING_URL=http://192.168.1.100:3003
```

Or via environment variable:
```bash
export MACHINE_LEARNING_URL=http://192.168.1.100:3003
```

**Important**: If your Immich is in Docker, use your Mac's LAN IP, not `localhost` or `127.0.0.1`.

## Running the Service

### Development Mode (with auto-reload):
```bash
source .venv/bin/activate
python -m src.main
```

### Production Mode (recommended):
```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 3003 --workers 1
```

**⚠️ Note**: Multi-worker mode (`--workers 2+`) is not recommended due to model loading behavior. Each worker would load its own copy of models into RAM.

### Running as a Service (macOS launchd):

Create `~/Library/LaunchAgents/com.immich.ml.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.immich.ml</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/your/.venv/bin/python</string>
        <string>-m</string>
        <string>src.main</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/immich-ml-metal</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/immich-ml.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/immich-ml-error.log</string>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.immich.ml.plist
launchctl start com.immich.ml
```

## Verification

Test the service is working:

```bash
# Health check
curl http://localhost:3003/ping
# Should return: pong

# Service info
curl http://localhost:3003/
# Should return: {"message":"Immich ML"}

# Test CLIP text encoding
curl -X POST http://localhost:3003/predict \
  -F 'entries={"clip":{"textual":{"modelName":"ViT-B-32__openai"}}}' \
  -F 'text=a photo of a cat'
```

In Immich, you should see the ML service connect in the admin logs.

## Troubleshooting

### Models not downloading
- Check internet connection
- Ensure `~/.cache/huggingface/` and `~/.insightface/` are writable
- Try manually: `huggingface-cli download mlx-community/clip-vit-base-patch32`

### "CoreML not available" warnings
- Normal for some models
- Service will fall back to CPU (slower but functional)
- Ensure macOS 13+ and Apple Silicon

### High memory usage
- Try smaller models (`ViT-B-32__openai`, `buffalo_s`)
- Limit concurrent Immich ML jobs in Immich settings
- Consider increasing Mac's swap space

### Face detection too aggressive/lenient
- Adjust `ML_FACE_MIN_SCORE` (lower = more faces detected)
- Values between 0.3-0.8 are typical
- Default config has inconsistencies - experiment to find what works

### Immich doesn't recognize the service
- Verify `MACHINE_LEARNING_URL` is correct
- Check firewall settings (port 3003 must be accessible)
- Ensure the service is actually running (`curl http://localhost:3003/ping`)
- Check Immich admin logs for connection errors

## Development

**Note**: Again, I'm not a developer - these were set up by Claude:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (basic tests exist, comprehensive testing needed)
pytest -v

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Known Issues & TODOs

See the [code review above](#) for detailed technical issues. Summary:

**Critical:**
- [ ] Configuration defaults inconsistency (face_min_score, face_model)
- [ ] Dependency mismatches between pyproject.toml and requirements.txt
- [ ] Verify embedding string format is correct for Immich
- [ ] Add proper logging infrastructure (currently uses print())
- [ ] Thread safety for model caching

**Important:**
- [ ] Comprehensive integration testing with real Immich
- [ ] Memory management for face recognition models
- [ ] Better error handling and fallbacks
- [ ] Add Docker support for easier deployment
- [ ] Resource limits (max upload size, timeouts)

**Nice to Have:**
- [ ] Async/await properly throughout
- [ ] Metrics/monitoring (Prometheus?)
- [ ] Configuration file support (YAML/TOML)
- [ ] Progress bars for model downloads
- [ ] Better health checks beyond /ping

## Contributing

Given this is primarily an AI-assisted project, contributions are **very welcome**, especially from actual developers who can:

- Review and improve the code quality
- Add proper tests
- Verify Immich compatibility  
- Optimize performance
- Add missing features
- Improve documentation

Please open issues for any problems you find!

## Performance Notes

In my testing on Mac Mini M4 (16GB RAM):

- **CLIP encoding**: ~100-300ms per image (ViT-B-32)
- **Face detection**: ~50-150ms per image (Vision framework on ANE)
- **Face recognition**: ~30-80ms per face (InsightFace with CoreML)
- **OCR**: ~100-400ms per image (Vision framework)

Your mileage will vary based on:
- Mac model (M1/M2/M3/M4, RAM amount)
- Image resolution
- Model choices
- System load

## Credits & Acknowledgments

- **Code Architecture & Implementation**: Claude (Anthropic) - AI assistant that wrote ~95% of this code
- **Requirements & Testing**: Sebastian (me) - provided specs, tested, gave feedback
- **Based on**: [Immich](https://immich.app/) by Alex Tran and community
- **Technologies**: 
  - [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
  - [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
  - [Apple Vision](https://developer.apple.com/documentation/vision) - Face detection & OCR
  - [open_clip](https://github.com/mlfoundations/open_clip) - CLIP fallback

## License

MIT License - See LICENSE file

## Support

This is a hobby project with no guarantees. That said:

- Open an issue for bugs or questions
- Check Immich discord #machine-learning channel for ML-related help  
- Be patient - I'm learning as I go!

## Final Notes

This project exists because I wanted to run Immich fully native on my Mac Mini without Docker Desktop overhead. Claude helped me understand the Immich ML API and implement it using Apple's native frameworks. 

**It works for me**, but may not work for everyone. Use at your own risk, and please contribute improvements if you can!

If you're an Immich core developer and this isn't quite right - sorry! I'm happy to take feedback and improve it, or if it's too messy, just archive it. The goal was to help the Mac community, not create more work for maintainers.

---

**tl;dr**: AI-written Immich ML service for Apple Silicon. Alpha quality. Use with caution. Contributions very welcome.