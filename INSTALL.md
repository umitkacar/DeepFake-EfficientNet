# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) CUDA-capable GPU for faster training/inference

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/umitkacar/DeepFake-EfficientNet.git
cd DeepFake-EfficientNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Method 2: Install from PyPI (When Available)

```bash
pip install deepfake-detector
```

### Method 3: Install from Wheel

```bash
# Build the package
python -m build

# Install the wheel
pip install dist/deepfake_detector-2.0.0-py3-none-any.whl
```

## Verify Installation

```python
python -c "import deepfake_detector; print(deepfake_detector.__version__)"
```

Expected output: `2.0.0`

## GPU Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### CUDA Out of Memory

Reduce batch size in training/inference:

```bash
python scripts/train.py --batch-size 16  # Instead of 32
```

### OpenCV Issues

If OpenCV import fails, try:

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

## Next Steps

- Read the [README](README.md) for usage examples
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
- Run example scripts in `scripts/` directory
