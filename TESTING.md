# 🧪 Testing Guide

This document provides comprehensive testing guidelines and examples for the Neural Network Breast Cancer Classification project.

## 📋 Table of Contents

- [Overview](#overview)
- [Testing Philosophy](#testing-philosophy)
- [Test Structure](#test-structure)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Model Tests](#model-tests)
- [Data Tests](#data-tests)
- [Performance Tests](#performance-tests)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)

## 🌟 Overview

Testing is crucial for ensuring the reliability and correctness of machine learning models. This project implements comprehensive testing strategies covering:

- **Data validation and preprocessing**
- **Model architecture and training**
- **Prediction accuracy and consistency**
- **Performance benchmarks**
- **Integration between components**

## 💭 Testing Philosophy

### Core Principles

1. **Reliability**: Ensure model predictions are consistent and accurate
2. **Reproducibility**: Tests should produce the same results across runs
3. **Maintainability**: Tests should be easy to understand and modify
4. **Comprehensive Coverage**: Test all critical paths and edge cases
5. **Fast Feedback**: Unit tests should run quickly for rapid development

### Testing Pyramid

```
                    /\
                   /  \
                  /    \
                 /  E2E  \     End-to-End Tests (Few)
                /________\
               /          \
              / Integration \   Integration Tests (Some)
             /______________\
            /                \
           /   Unit Tests      \  Unit Tests (Many)
          /____________________\
```

## 📁 Test Structure

```
tests/
├── 📄 __init__.py
├── 📄 conftest.py                    # Pytest configuration and fixtures
├── 📁 unit/                          # Unit tests
│   ├── 📄 test_data_processing.py
│   ├── 📄 test_model_architecture.py
│   ├── 📄 test_training.py
│   ├── 📄 test_evaluation.py
│   └── 📄 test_utilities.py
├── 📁 integration/                   # Integration tests
│   ├── 📄 test_training_pipeline.py
│   ├── 📄 test_data_pipeline.py
│   └── 📄 test_model_serving.py
├── 📁 data/                          # Test data
│   ├── 📄 sample_dataset.csv
│   └── 📄 test_cases.json
└── 📁 fixtures/                      # Test fixtures
    ├── 📄 models.py
    └── 📄 data.py
```

## 🔬 Unit Tests

### Data Processing Tests

```python
# tests/unit/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_dataset, preprocess_data, scale_features

class TestDataProcessing:

    def test_load_dataset_success(self, sample_dataset_path):
        """Test successful dataset loading."""
        data = load_dataset(sample_dataset_path)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'diagnosis' in data.columns

    def test_load_dataset_file_not_found(self):
        """Test handling of missing dataset file."""
        with pytest.raises(FileNotFoundError):
            load_dataset('nonexistent_file.csv')

    def test_preprocess_data_with_diagnosis(self, sample_data):
        """Test preprocessing with diagnosis column."""
        X, y = preprocess_data(sample_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert 'diagnosis' not in X.columns
        assert 'id' not in X.columns
        assert len(X) == len(y)
        assert y.dtype == int
        assert set(y.unique()) <= {0, 1}

    def test_preprocess_data_without_diagnosis(self, sample_features):
        """Test preprocessing without diagnosis column."""
        X, y = preprocess_data(sample_features)

        assert isinstance(X, pd.DataFrame)
        assert y is None
        assert 'id' not in X.columns

    def test_scale_features_train_only(self, sample_features_array):
        """Test feature scaling with training data only."""
        X_scaled, scaler = scale_features(sample_features_array)

        assert X_scaled.shape == sample_features_array.shape
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)

    def test_scale_features_train_test(self, sample_features_array):
        """Test feature scaling with training and test data."""
        X_train, X_test = sample_features_array[:80], sample_features_array[80:]
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
```

### Model Architecture Tests

```python
# tests/unit/test_model_architecture.py
import pytest
import tensorflow as tf
from src.model_architecture import create_model, get_model_summary

class TestModelArchitecture:

    def test_create_model_default_params(self):
        """Test model creation with default parameters."""
        model = create_model(input_dim=30)

        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 30)
        assert model.output_shape == (None, 1)
        assert len(model.layers) > 0

    def test_create_model_custom_params(self):
        """Test model creation with custom parameters."""
        hidden_layers = [64, 32]
        dropout_rate = 0.5

        model = create_model(
            input_dim=30,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        assert model.input_shape == (None, 30)
        assert model.output_shape == (None, 1)

        # Count dense layers (excluding input and output)
        dense_layers = [layer for layer in model.layers
                       if isinstance(layer, tf.keras.layers.Dense)]
        assert len(dense_layers) == len(hidden_layers) + 1  # +1 for output layer

    def test_model_compilation(self):
        """Test that model is properly compiled."""
        model = create_model(input_dim=30)

        assert model.optimizer is not None
        assert model.loss is not None
        assert 'accuracy' in model.metrics_names

    def test_model_prediction_shape(self):
        """Test model prediction output shape."""
        model = create_model(input_dim=30)
        X_dummy = tf.random.normal((10, 30))

        predictions = model.predict(X_dummy, verbose=0)

        assert predictions.shape == (10, 1)
        assert np.all((predictions >= 0) & (predictions <= 1))  # Sigmoid output

    @pytest.mark.parametrize("input_dim,expected_shape", [
        (20, (None, 20)),
        (30, (None, 30)),
        (40, (None, 40))
    ])
    def test_model_input_dimensions(self, input_dim, expected_shape):
        """Test model with different input dimensions."""
        model = create_model(input_dim=input_dim)
        assert model.input_shape == expected_shape
```

### Training Tests

```python
# tests/unit/test_training.py
import pytest
import numpy as np
import tensorflow as tf
from src.training import train_model, create_callbacks

class TestTraining:

    def test_train_model_basic(self, trained_model, sample_data_scaled):
        """Test basic model training functionality."""
        X_train, y_train = sample_data_scaled

        history = train_model(
            trained_model, X_train, y_train,
            epochs=2, verbose=0  # Quick test
        )

        assert hasattr(history, 'history')
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 2

    def test_train_model_with_validation(self, trained_model, sample_data_scaled):
        """Test training with validation split."""
        X_train, y_train = sample_data_scaled

        history = train_model(
            trained_model, X_train, y_train,
            validation_split=0.2, epochs=2, verbose=0
        )

        assert 'val_loss' in history.history
        assert 'val_accuracy' in history.history

    def test_create_callbacks(self, tmp_path):
        """Test callback creation."""
        checkpoint_path = str(tmp_path / "model_checkpoint.h5")
        callbacks = create_callbacks(checkpoint_path)

        assert len(callbacks) > 0
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types
        assert 'ReduceLROnPlateau' in callback_types

    def test_training_improves_loss(self, trained_model, sample_data_scaled):
        """Test that training actually improves the model."""
        X_train, y_train = sample_data_scaled

        # Get initial loss
        initial_loss = trained_model.evaluate(X_train, y_train, verbose=0)[0]

        # Train model
        train_model(trained_model, X_train, y_train, epochs=5, verbose=0)

        # Get final loss
        final_loss = trained_model.evaluate(X_train, y_train, verbose=0)[0]

        assert final_loss < initial_loss, "Training should improve loss"
```

## 🔗 Integration Tests

### Training Pipeline Tests

```python
# tests/integration/test_training_pipeline.py
import pytest
import numpy as np
from src.pipeline import BreastCancerPipeline

class TestTrainingPipeline:

    def test_complete_training_pipeline(self, sample_dataset_path):
        """Test complete training pipeline from data to model."""
        pipeline = BreastCancerPipeline()

        # Load and preprocess data
        pipeline.load_data(sample_dataset_path)
        pipeline.preprocess_data()
        pipeline.split_data(test_size=0.2, random_state=42)

        # Train model
        model = pipeline.create_model()
        history = pipeline.train_model(epochs=10, verbose=0)

        # Evaluate model
        metrics = pipeline.evaluate_model()

        # Assertions
        assert model is not None
        assert history is not None
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0.5  # Should be better than random

    def test_pipeline_reproducibility(self, sample_dataset_path):
        """Test that pipeline produces reproducible results."""
        # First run
        pipeline1 = BreastCancerPipeline(random_state=42)
        pipeline1.load_data(sample_dataset_path)
        pipeline1.preprocess_data()
        pipeline1.split_data()
        model1 = pipeline1.create_model()
        history1 = pipeline1.train_model(epochs=5, verbose=0)

        # Second run with same random state
        pipeline2 = BreastCancerPipeline(random_state=42)
        pipeline2.load_data(sample_dataset_path)
        pipeline2.preprocess_data()
        pipeline2.split_data()
        model2 = pipeline2.create_model()
        history2 = pipeline2.train_model(epochs=5, verbose=0)

        # Compare final losses (should be close due to randomness in training)
        loss1 = history1.history['loss'][-1]
        loss2 = history2.history['loss'][-1]
        assert abs(loss1 - loss2) < 0.1, "Results should be reproducible"
```

## 🤖 Model Tests

### Model Performance Tests

```python
# tests/unit/test_model_performance.py
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

class TestModelPerformance:

    def test_model_accuracy_threshold(self, trained_model, test_data):
        """Test that model meets minimum accuracy threshold."""
        X_test, y_test = test_data

        predictions = trained_model.predict(X_test, verbose=0)
        y_pred = (predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.85, f"Model accuracy {accuracy:.3f} below threshold"

    def test_model_precision_recall(self, trained_model, test_data):
        """Test model precision and recall."""
        X_test, y_test = test_data

        predictions = trained_model.predict(X_test, verbose=0)
        y_pred = (predictions > 0.5).astype(int).flatten()

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        assert precision > 0.8, f"Precision {precision:.3f} too low"
        assert recall > 0.8, f"Recall {recall:.3f} too low"

    def test_prediction_consistency(self, trained_model, test_data):
        """Test that model predictions are consistent."""
        X_test, _ = test_data

        # Make predictions multiple times
        pred1 = trained_model.predict(X_test, verbose=0)
        pred2 = trained_model.predict(X_test, verbose=0)

        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)

    def test_model_confidence_distribution(self, trained_model, test_data):
        """Test that model produces reasonable confidence distribution."""
        X_test, _ = test_data

        predictions = trained_model.predict(X_test, verbose=0).flatten()

        # Check that predictions span reasonable range
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0
        assert predictions.std() > 0.1  # Should have some variation
```

## 📊 Data Tests

### Data Validation Tests

```python
# tests/unit/test_data_validation.py
import pytest
import pandas as pd
import numpy as np

class TestDataValidation:

    def test_dataset_structure(self, sample_data):
        """Test that dataset has expected structure."""
        assert isinstance(sample_data, pd.DataFrame)
        assert not sample_data.empty
        assert len(sample_data.columns) >= 31  # 30 features + diagnosis

    def test_feature_data_types(self, sample_data):
        """Test that features have correct data types."""
        feature_columns = [col for col in sample_data.columns
                          if col not in ['id', 'diagnosis']]

        for col in feature_columns:
            assert pd.api.types.is_numeric_dtype(sample_data[col]), f"{col} should be numeric"

    def test_no_missing_values(self, sample_data):
        """Test that dataset has no missing values."""
        missing_counts = sample_data.isnull().sum()
        assert missing_counts.sum() == 0, "Dataset should have no missing values"

    def test_target_variable_binary(self, sample_data):
        """Test that target variable is binary."""
        if 'diagnosis' in sample_data.columns:
            unique_values = sample_data['diagnosis'].unique()
            assert len(unique_values) == 2, "Target should be binary"
            assert set(unique_values) <= {'M', 'B'}, "Target values should be M or B"

    def test_feature_ranges(self, sample_data):
        """Test that features are within reasonable ranges."""
        feature_columns = [col for col in sample_data.columns
                          if col not in ['id', 'diagnosis']]

        for col in feature_columns:
            values = sample_data[col]
            assert values.min() >= 0, f"{col} should be non-negative"
            assert np.isfinite(values).all(), f"{col} should contain no infinite values"
```

## 🏃 Running Tests

### Command Line Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_data_processing.py

# Run specific test class
pytest tests/unit/test_model_architecture.py::TestModelArchitecture

# Run specific test method
pytest tests/unit/test_training.py::TestTraining::test_train_model_basic

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto

# Run only fast tests (marked with @pytest.mark.fast)
pytest -m fast

# Run tests and generate XML report
pytest --junitxml=test-results.xml
```

### Configuration File

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    slow: marks tests as slow
    fast: marks tests as fast
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    gpu: marks tests that require GPU
```

## 📈 Test Coverage

### Coverage Goals

- **Overall Coverage**: > 90%
- **Critical Functions**: 100%
- **Model Training**: > 95%
- **Data Processing**: > 95%

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate terminal coverage report
pytest --cov=src --cov-report=term

# Generate XML coverage report (for CI)
pytest --cov=src --cov-report=xml
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = src
omit =
    */tests/*
    */venv/*
    */__init__.py
    */setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## 🔄 Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v1
        with:
          file: ./coverage.xml
```

## 🔧 Test Fixtures

### Pytest Fixtures

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

@pytest.fixture
def sample_dataset_path():
    """Path to sample dataset for testing."""
    return Path(__file__).parent / "data" / "sample_dataset.csv"

@pytest.fixture
def sample_data():
    """Sample breast cancer dataset."""
    np.random.seed(42)
    n_samples = 100

    # Generate synthetic features
    data = {}
    feature_names = [f'feature_{i}' for i in range(30)]

    for name in feature_names:
        data[name] = np.random.normal(0, 1, n_samples)

    # Add target variable
    data['diagnosis'] = np.random.choice(['M', 'B'], n_samples)
    data['id'] = range(n_samples)

    return pd.DataFrame(data)

@pytest.fixture
def trained_model():
    """Pre-trained model for testing."""
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

@pytest.fixture
def sample_data_scaled():
    """Scaled sample data for training tests."""
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 30))
    y = np.random.randint(0, 2, 100)
    return X, y
```

## 🐛 Debugging Tests

### Common Issues and Solutions

1. **TensorFlow GPU Issues**:

   ```python
   # Use CPU only for tests
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
   ```

2. **Random Seed Issues**:

   ```python
   # Set all random seeds
   import random
   import numpy as np
   import tensorflow as tf

   random.seed(42)
   np.random.seed(42)
   tf.random.set_seed(42)
   ```

3. **Memory Issues**:
   ```python
   # Clear session between tests
   @pytest.fixture(autouse=True)
   def clear_session():
       yield
       tf.keras.backend.clear_session()
   ```

## 📝 Best Practices

### Writing Good Tests

1. **Descriptive Names**: Use clear, descriptive test names
2. **Single Responsibility**: Each test should test one thing
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Tests**: Unit tests should run quickly
6. **Realistic Data**: Use realistic test data when possible

### Test Organization

1. **Group Related Tests**: Use test classes to group related tests
2. **Use Fixtures**: Share setup code with fixtures
3. **Mark Tests**: Use pytest markers to categorize tests
4. **Document Tests**: Add docstrings to explain complex tests

---

_This testing guide ensures robust, reliable machine learning models through comprehensive testing strategies._
