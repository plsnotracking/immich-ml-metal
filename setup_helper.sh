#!/bin/bash
# Setup helper for immich-ml-metal
# Helps pin mlx-clip to current commit and install dependencies

set -e

echo "=================================="
echo "immich-ml-metal Setup Helper"
echo "=================================="
echo

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    echo "   Run this script from the immich-ml-metal directory"
    exit 1
fi

# Get current mlx-clip commit hash
echo "🔍 Fetching current mlx-clip commit hash..."
COMMIT_HASH=$(git ls-remote https://github.com/harperreed/mlx_clip.git HEAD | cut -f1)

if [ -z "$COMMIT_HASH" ]; then
    echo "❌ Error: Could not fetch commit hash"
    echo "   Check your internet connection and try again"
    exit 1
fi

echo "✅ Found commit: $COMMIT_HASH"
echo

# Check if requirements.txt needs updating
if grep -q "COMMIT_HASH" requirements.txt; then
    echo "📝 Updating requirements.txt with commit hash..."
    
    # Create backup
    cp requirements.txt requirements.txt.backup
    echo "   Created backup: requirements.txt.backup"
    
    # Update the file
    sed -i.tmp "s/COMMIT_HASH/$COMMIT_HASH/g" requirements.txt
    rm -f requirements.txt.tmp
    
    echo "✅ Updated requirements.txt"
    echo
else
    echo "ℹ️  requirements.txt already has a commit hash"
    echo "   Current line:"
    grep "mlx-clip @" requirements.txt
    echo
    read -p "   Replace with new commit? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp requirements.txt requirements.txt.backup
        sed -i.tmp "s/@[a-f0-9]\{40\}/@$COMMIT_HASH/g" requirements.txt
        rm -f requirements.txt.tmp
        echo "✅ Updated to new commit"
    else
        echo "   Keeping existing commit hash"
    fi
    echo
fi

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
    echo
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies (this may take a few minutes)..."
echo "   This will download models on first run (~500MB-2GB)"
echo

pip install --upgrade pip
pip install -r requirements.txt

echo
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo
echo "2. Start the service:"
echo "   python -m src.main"
echo
echo "3. Test the service:"
echo "   curl http://localhost:3003/ping"
echo
echo "4. Configure Immich to use this service:"
echo "   MACHINE_LEARNING_URL=http://YOUR_MAC_IP:3003"
echo
echo "Configuration:"
echo "  - CLIP model: $ML_CLIP_MODEL (default: ViT-B-32__openai)"
echo "  - Face model: $ML_FACE_MODEL (default: buffalo_l)"  
echo "  - Face threshold: $ML_FACE_MIN_SCORE (default: 0.7)"
echo
echo "See README.md for more configuration options"