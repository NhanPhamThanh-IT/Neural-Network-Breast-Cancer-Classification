# 🔌 API Documentation

This document provides comprehensive API documentation for the Neural Network Breast Cancer Classification project, including function signatures, parameters, return values, and usage examples.

## 📋 Table of Contents

- [Overview](#overview)
- [Data Processing API](#data-processing-api)
- [Model Architecture API](#model-architecture-api)
- [Training API](#training-api)
- [Evaluation API](#evaluation-api)
- [Visualization API](#visualization-api)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Examples](#examples)

## 🌟 Overview

The project follows a modular design with clear separation of concerns:

- **Data Processing**: Loading, cleaning, and preprocessing functions
- **Model Architecture**: Neural network definition and configuration
- **Training**: Model training, validation, and optimization
- **Evaluation**: Performance metrics and model assessment
- **Visualization**: Plotting and analysis visualization
- **Utilities**: Helper functions and common operations

## 📊 Data Processing API

### `load_dataset(file_path: str) -> pd.DataFrame`

Load the Wisconsin Breast Cancer Dataset from CSV file.

**Parameters:**

- `file_path` (str): Path to the CSV dataset file

**Returns:**

- `pd.DataFrame`: Raw dataset with all original columns

**Example:**

```python
import pandas as pd

def load_dataset(file_path):
    """
    Load breast cancer dataset from CSV file.

    Args:
        file_path (str): Path to dataset CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Dataset file is empty")

# Usage
dataset = load_dataset('models/breast_cancer_dataset.csv')
```

### `preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]`

Preprocess the dataset for training.

**Parameters:**

- `data` (pd.DataFrame): Raw dataset

**Returns:**

- `Tuple[pd.DataFrame, pd.Series]`: Features (X) and target (y)

**Example:**

```python
def preprocess_data(data):
    """
    Preprocess breast cancer dataset.

    Args:
        data (pd.DataFrame): Raw dataset

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features and target
    """
    # Remove ID column
    data_processed = data.drop(['id'], axis=1, errors='ignore')

    # Separate features and target
    if 'diagnosis' in data_processed.columns:
        X = data_processed.drop(['diagnosis'], axis=1)
        y = data_processed['diagnosis']

        # Convert target to binary (M=1, B=0)
        y = (y == 'M').astype(int)
    else:
        X = data_processed
        y = None

    return X, y
```

### `scale_features(X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, ...]`

Scale features using StandardScaler.

**Parameters:**

- `X_train` (np.ndarray): Training features
- `X_test` (np.ndarray, optional): Test features

**Returns:**

- `Tuple`: Scaled training features, (scaled test features), scaler object

**Example:**

```python
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test=None):
    """
    Scale features using StandardScaler.

    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray, optional): Test features

    Returns:
        Tuple: Scaled data and scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler
```

## 🧠 Model Architecture API

### `create_model(input_dim: int, **kwargs) -> tf.keras.Model`

Create and configure the neural network model.

**Parameters:**

- `input_dim` (int): Number of input features
- `hidden_layers` (list, optional): Hidden layer sizes. Default: [128, 64, 32]
- `dropout_rate` (float, optional): Dropout rate. Default: 0.3
- `activation` (str, optional): Activation function. Default: 'relu'
- `use_batch_norm` (bool, optional): Use batch normalization. Default: True

**Returns:**

- `tf.keras.Model`: Compiled neural network model

**Example:**

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def create_model(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3,
                activation='relu', use_batch_norm=True):
    """
    Create neural network model for breast cancer classification.

    Args:
        input_dim (int): Number of input features
        hidden_layers (list): Hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
        activation (str): Activation function for hidden layers
        use_batch_norm (bool): Whether to use batch normalization

    Returns:
        tf.keras.Model: Compiled model
    """
    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='input_features')
    x = inputs

    # Hidden layers
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation=activation, name=f'dense_{i+1}')(x)

        if use_batch_norm:
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='breast_cancer_classifier')

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model
```

### `get_model_summary(model: tf.keras.Model) -> str`

Get detailed model architecture summary.

**Parameters:**

- `model` (tf.keras.Model): Keras model

**Returns:**

- `str`: Model summary string

## 🏋️ Training API

### `train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> tf.keras.callbacks.History`

Train the neural network model.

**Parameters:**

- `model` (tf.keras.Model): Model to train
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training labels
- `validation_split` (float, optional): Validation split ratio. Default: 0.2
- `epochs` (int, optional): Number of training epochs. Default: 100
- `batch_size` (int, optional): Training batch size. Default: 32
- `callbacks` (list, optional): Keras callbacks

**Returns:**

- `tf.keras.callbacks.History`: Training history

**Example:**

```python
def train_model(model, X_train, y_train, validation_split=0.2, epochs=100,
               batch_size=32, callbacks=None, verbose=1):
    """
    Train the neural network model.

    Args:
        model (tf.keras.Model): Model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        validation_split (float): Validation data split
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        callbacks (list): Keras callbacks
        verbose (int): Verbosity mode

    Returns:
        tf.keras.callbacks.History: Training history
    """
    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]

    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    return history
```

### `create_callbacks(model_checkpoint_path: str = None) -> List[tf.keras.callbacks.Callback]`

Create standard training callbacks.

**Parameters:**

- `model_checkpoint_path` (str, optional): Path to save model checkpoints

**Returns:**

- `List[tf.keras.callbacks.Callback]`: List of configured callbacks

## 📈 Evaluation API

### `evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]`

Comprehensive model evaluation.

**Parameters:**

- `model` (tf.keras.Model): Trained model
- `X_test` (np.ndarray): Test features
- `y_test` (np.ndarray): Test labels

**Returns:**

- `Dict[str, float]`: Dictionary of evaluation metrics

**Example:**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with comprehensive metrics.

    Args:
        model (tf.keras.Model): Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    return metrics
```

### `confusion_matrix_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[np.ndarray, float]]`

Detailed confusion matrix analysis.

**Parameters:**

- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels

**Returns:**

- `Dict`: Confusion matrix and derived metrics

## 📊 Visualization API

### `plot_training_history(history: tf.keras.callbacks.History) -> plt.Figure`

Plot training and validation metrics over epochs.

**Parameters:**

- `history` (tf.keras.callbacks.History): Training history

**Returns:**

- `plt.Figure`: Matplotlib figure object

**Example:**

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plot training history metrics.

    Args:
        history (tf.keras.callbacks.History): Training history

    Returns:
        plt.Figure: Training plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Additional metrics can be added

    plt.tight_layout()
    return fig
```

### `plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure`

Plot confusion matrix heatmap.

### `plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> plt.Figure`

Plot ROC curve with AUC score.

### `plot_feature_importance(model: tf.keras.Model, feature_names: List[str]) -> plt.Figure`

Plot feature importance analysis.

## 🛠️ Utility Functions

### `save_model(model: tf.keras.Model, filepath: str) -> None`

Save trained model to disk.

**Parameters:**

- `model` (tf.keras.Model): Model to save
- `filepath` (str): Path to save model

### `load_model(filepath: str) -> tf.keras.Model`

Load saved model from disk.

**Parameters:**

- `filepath` (str): Path to saved model

**Returns:**

- `tf.keras.Model`: Loaded model

### `get_feature_names() -> List[str]`

Get standard feature names for the breast cancer dataset.

**Returns:**

- `List[str]`: List of feature names

## ⚙️ Configuration

### `ModelConfig`

Configuration class for model parameters.

```python
class ModelConfig:
    """Configuration for neural network model."""

    def __init__(self):
        self.input_dim = 30
        self.hidden_layers = [128, 64, 32]
        self.dropout_rate = 0.3
        self.activation = 'relu'
        self.use_batch_norm = True
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        self.validation_split = 0.2
```

### `TrainingConfig`

Configuration class for training parameters.

## 💡 Examples

### Complete Training Pipeline

```python
# Load and preprocess data
data = load_dataset('models/breast_cancer_dataset.csv')
X, y = preprocess_data(data)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Create and train model
model = create_model(input_dim=X_train_scaled.shape[1])
history = train_model(model, X_train_scaled, y_train)

# Evaluate model
metrics = evaluate_model(model, X_test_scaled, y_test)
print("Model Performance:", metrics)

# Visualize results
plot_training_history(history)
plot_confusion_matrix(y_test, model.predict(X_test_scaled) > 0.5)

# Save model
save_model(model, 'models/saved_models/breast_cancer_model.h5')
```

### Custom Model Configuration

```python
# Custom configuration
config = ModelConfig()
config.hidden_layers = [256, 128, 64, 32]
config.dropout_rate = 0.4
config.epochs = 150

# Create model with custom config
model = create_model(
    input_dim=30,
    hidden_layers=config.hidden_layers,
    dropout_rate=config.dropout_rate
)
```

## 🔗 Dependencies

Key dependencies used in the API:

- **TensorFlow/Keras**: Neural network implementation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization
- **SciPy**: Statistical functions

## 📝 Error Handling

All functions include appropriate error handling:

- **Input validation**: Check parameter types and ranges
- **File operations**: Handle file not found and permission errors
- **Model operations**: Handle training failures and invalid models
- **Data validation**: Check data shapes and types

## 🔄 Version Compatibility

- **TensorFlow**: 2.8+ (supports tf.keras)
- **Python**: 3.8+ (f-strings and type hints)
- **NumPy**: 1.21+ (array compatibility)
- **Pandas**: 1.3+ (DataFrame methods)

---

_This API documentation is maintained alongside the codebase and updated with each release._
