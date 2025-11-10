# Contributing to DeepFake-EfficientNet

Thank you for your interest in contributing to DeepFake-EfficientNet! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

### Development Setup

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/DeepFake-EfficientNet.git
cd DeepFake-EfficientNet
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks:**

```bash
pre-commit install
```

## 🔨 Development Workflow

### Before You Start

1. Create a new branch for your feature/fix:
```bash
git checkout -b feature/your-feature-name
```

2. Make sure tests pass:
```bash
make test
```

### Making Changes

1. **Write code** following our style guidelines (see below)
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run quality checks:**

```bash
make format  # Format code
make lint    # Run linters
make test    # Run tests
```

### Submitting Changes

1. **Commit your changes:**

```bash
git add .
git commit -m "feat: add amazing feature"
```

*Note: Pre-commit hooks will run automatically*

2. **Push to your fork:**

```bash
git push origin feature/your-feature-name
```

3. **Create a Pull Request** on GitHub

## 📝 Code Style Guidelines

### Python Style

- **PEP 8** compliant (enforced by Black and Ruff)
- **Line length:** 100 characters
- **Type hints:** Use type hints for function signatures
- **Docstrings:** Google style for all public functions/classes

Example:

```python
def detect_deepfake(
    image: np.ndarray,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Detect if an image is a deepfake.

    Args:
        image: Input image as numpy array (H, W, C)
        threshold: Classification threshold

    Returns:
        Tuple of (is_fake, confidence)

    Example:
        >>> result, confidence = detect_deepfake(img)
        >>> print(f"Fake: {result}, Confidence: {confidence:.2f}")
    """
    # Implementation
    pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add EfficientNet-B7 support
fix: resolve MTCNN memory leak
docs: update installation instructions
test: add tests for metrics module
```

## 🧪 Testing

### Writing Tests

- Place tests in `tests/` directory
- Use pytest fixtures from `tests/conftest.py`
- Aim for >80% code coverage

Example test:

```python
def test_model_inference(sample_image):
    """Test model can perform inference."""
    model = DeepFakeDetector('efficientnet-b0')
    output = model(sample_image)
    assert output.shape == (1, 2)
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=deepfake_detector

# Specific test
pytest tests/unit/test_model.py

# In parallel
pytest -n auto
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """
    Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is invalid

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
```

### Updating README

When adding new features, update:
- Feature list
- Usage examples
- Installation instructions (if dependencies change)

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description:** Clear description of the bug
2. **Steps to reproduce:** Minimal code example
3. **Expected behavior:** What should happen
4. **Actual behavior:** What actually happens
5. **Environment:**
   - Python version
   - OS
   - Package versions

## 💡 Feature Requests

For feature requests:

1. **Use case:** Explain why this feature is needed
2. **Proposed solution:** How you envision it working
3. **Alternatives:** Other approaches you considered

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (Black, Ruff pass)
- [ ] Type hints added for new code
- [ ] Docstrings added/updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts

## 🔍 Code Review Process

1. **Automated checks** must pass (linting, tests)
2. **Maintainer review** for code quality and design
3. **Address feedback** and update PR
4. **Approval** and merge by maintainer

## 🎯 Areas for Contribution

Looking to contribute but not sure where? Check out:

- **Good first issues:** Tagged on GitHub
- **Documentation:** Always room for improvement
- **Tests:** Increase coverage
- **Examples:** Add usage examples
- **Performance:** Optimize inference speed
- **Features:** Implement SOTA detection methods

## 📖 Resources

- [Project README](README.md)
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)

## 💬 Communication

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, ideas
- **Pull Requests:** Code contributions

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DeepFake-EfficientNet! 🙏
