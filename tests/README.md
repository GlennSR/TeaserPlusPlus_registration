# Tests

This directory contains unit tests for the point cloud registration project.

## Running Tests

### With uv (Recommended)

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest

# Run with coverage report (generated in reports/ directory)
uv run pytest

# Run specific test file
uv run pytest tests/test_common.py

# Run specific test
uv run pytest tests/test_common.py::TestRotationError::test_identical_rotations

# Run with verbose output
uv run pytest -v

# Run with print statements shown
uv run pytest -s
```

### With pip/venv

```bash
# Activate virtual environment
source .venv/bin/activate

# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage (reports generated in reports/ directory)
pytest
```

## Test Structure

- `conftest.py`: Shared fixtures and test configuration
- `test_common.py`: Tests for utility functions in `common.py`
- Add more test files as needed following the `test_*.py` naming convention

## Writing Tests

Example test structure:

```python
import pytest
from your_module import your_function

class TestYourFunction:
    """Tests for your_function."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = your_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

## Coverage Reports

After running tests with coverage, reports are generated in the `reports/` directory:

- **HTML Report**: `reports/coverage/index.html` - Detailed line-by-line coverage
- **XML Report**: `reports/coverage.xml` - For CI/CD tools like Codecov
- **JSON Report**: `reports/coverage.json` - Machine-readable format

```bash
# Generate and open coverage report
uv run pytest
open reports/coverage/index.html  # macOS
# or
xdg-open reports/coverage/index.html  # Linux
```
