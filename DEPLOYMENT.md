# 🚀 Deployment Guide

This comprehensive guide covers various deployment strategies for the Neural Network Breast Cancer Classification model, from local serving to cloud deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Local Deployment](#local-deployment)
- [Web Application Deployment](#web-application-deployment)
- [API Deployment](#api-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Docker Deployment](#docker-deployment)
- [Model Serving](#model-serving)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Security Considerations](#security-considerations)

## 🌟 Overview

The breast cancer classification model can be deployed in various ways depending on your use case:

- **Local Desktop Application**: For individual use or testing
- **Web Application**: User-friendly interface for medical professionals
- **REST API**: Integration with existing medical systems
- **Cloud Platform**: Scalable deployment for production use
- **Mobile Application**: Portable diagnostic tool
- **Batch Processing**: Large-scale data processing

## 💻 Local Deployment

### Jupyter Notebook Interface

The simplest deployment method for research and development.

```python
# local_interface.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained breast cancer classification model."""
    model = tf.keras.models.load_model('models/saved_models/breast_cancer_model.h5')
    return model

@st.cache_data
def load_scaler():
    """Load the feature scaler."""
    import joblib
    scaler = joblib.load('models/saved_models/feature_scaler.pkl')
    return scaler

def main():
    st.title("🏥 Breast Cancer Classification Tool")
    st.markdown("Advanced neural network for breast cancer diagnosis support")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                               ["Prediction", "Model Information", "Feature Analysis"])

    if page == "Prediction":
        prediction_page()
    elif page == "Model Information":
        model_info_page()
    else:
        feature_analysis_page()

def prediction_page():
    st.header("🔬 Breast Cancer Prediction")

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()

    # Input methods
    input_method = st.radio("Choose input method:",
                           ["Manual Input", "Upload CSV", "Use Sample Data"])

    if input_method == "Manual Input":
        features = manual_input()
    elif input_method == "Upload CSV":
        features = csv_upload()
    else:
        features = sample_data()

    if features is not None:
        # Make prediction
        if st.button("🔍 Analyze Sample", type="primary"):
            prediction, confidence = make_prediction(model, scaler, features)
            display_results(prediction, confidence, features)

def manual_input():
    """Create manual input form for features."""
    st.subheader("Enter Cell Nucleus Measurements")

    col1, col2, col3 = st.columns(3)

    features = {}

    # Key features with descriptions
    key_features = {
        'radius_mean': 'Mean radius of cell nuclei',
        'texture_mean': 'Mean texture (gray-scale variation)',
        'perimeter_mean': 'Mean perimeter of cell nuclei',
        'area_mean': 'Mean area of cell nuclei',
        'smoothness_mean': 'Mean smoothness (local radius variation)',
        'compactness_mean': 'Mean compactness (perimeter²/area - 1)',
    }

    with col1:
        st.markdown("**Size Measurements**")
        features['radius_mean'] = st.number_input("Radius (mean)", 0.0, 50.0, 14.0, help=key_features['radius_mean'])
        features['perimeter_mean'] = st.number_input("Perimeter (mean)", 0.0, 200.0, 90.0)
        features['area_mean'] = st.number_input("Area (mean)", 0.0, 2500.0, 650.0)

    with col2:
        st.markdown("**Shape Measurements**")
        features['texture_mean'] = st.number_input("Texture (mean)", 0.0, 50.0, 19.0, help=key_features['texture_mean'])
        features['smoothness_mean'] = st.number_input("Smoothness (mean)", 0.0, 0.2, 0.1)
        features['compactness_mean'] = st.number_input("Compactness (mean)", 0.0, 0.5, 0.1)

    # Add remaining features with default values
    remaining_features = [
        'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
        'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    # Set default values for remaining features
    for feature in remaining_features:
        features[feature] = 0.1  # Default value

    return pd.DataFrame([features])

def make_prediction(model, scaler, features):
    """Make prediction using the trained model."""
    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
    prediction = "Malignant" if prediction_proba > 0.5 else "Benign"
    confidence = prediction_proba if prediction_proba > 0.5 else 1 - prediction_proba

    return prediction, confidence

def display_results(prediction, confidence, features):
    """Display prediction results with visualizations."""
    st.markdown("---")
    st.header("📊 Analysis Results")

    # Main result
    col1, col2 = st.columns(2)

    with col1:
        if prediction == "Malignant":
            st.error(f"⚠️ **Prediction: {prediction}**")
        else:
            st.success(f"✅ **Prediction: {prediction}**")

    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    # Confidence gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature importance (simplified)
    st.subheader("🔍 Key Feature Analysis")

    # Display top features
    feature_importance = {
        'radius_mean': features['radius_mean'].iloc[0],
        'texture_mean': features['texture_mean'].iloc[0],
        'perimeter_mean': features['perimeter_mean'].iloc[0],
        'area_mean': features['area_mean'].iloc[0],
    }

    fig_bar = px.bar(
        x=list(feature_importance.keys()),
        y=list(feature_importance.values()),
        title="Key Feature Values",
        labels={'x': 'Features', 'y': 'Values'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Medical disclaimer
    st.warning("""
    ⚕️ **Medical Disclaimer**:
    This tool is for educational and research purposes only.
    It should not be used as a substitute for professional medical advice,
    diagnosis, or treatment. Always consult with qualified healthcare professionals.
    """)

if __name__ == "__main__":
    main()
```

### Running the Local Interface

```bash
# Install Streamlit
pip install streamlit plotly

# Run the application
streamlit run local_interface.py
```

## 🌐 Web Application Deployment

### Flask REST API

```python
# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler at startup
try:
    model = tf.keras.models.load_model('models/saved_models/breast_cancer_model.h5')
    scaler = joblib.load('models/saved_models/feature_scaler.pkl')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    scaler = None

@app.route('/')
def home():
    """Serve the main interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions via API endpoint."""
    try:
        # Get JSON data
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        # Validate model availability
        if model is None or scaler is None:
            return jsonify({'error': 'Model not available'}), 503

        # Prepare features
        features = np.array(data['features']).reshape(1, -1)

        # Validate feature count
        if features.shape[1] != 30:
            return jsonify({'error': f'Expected 30 features, got {features.shape[1]}'}), 400

        # Scale features and make prediction
        features_scaled = scaler.transform(features)
        prediction_proba = model.predict(features_scaled, verbose=0)[0][0]

        # Prepare response
        result = {
            'prediction': 'Malignant' if prediction_proba > 0.5 else 'Benign',
            'probability': float(prediction_proba),
            'confidence': float(max(prediction_proba, 1 - prediction_proba)),
            'risk_level': get_risk_level(prediction_proba)
        }

        logger.info(f"Prediction made: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy' if model is not None and scaler is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'version': '1.0.0'
    }
    return jsonify(status)

def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.7:
        return 'Medium'
    else:
        return 'High'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### HTML Template

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Breast Cancer Classification</title>
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
      rel="stylesheet"
    />
    <style>
      .result-card {
        margin-top: 20px;
      }
      .malignant {
        border-left: 5px solid #dc3545;
      }
      .benign {
        border-left: 5px solid #28a745;
      }
      .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
      }
    </style>
  </head>
  <body>
    <div class="container mt-4">
      <div class="row justify-content-center">
        <div class="col-md-8">
          <h1 class="text-center mb-4">🏥 Breast Cancer Classification Tool</h1>

          <div class="card">
            <div class="card-body">
              <h5 class="card-title">Enter Cell Nucleus Measurements</h5>
              <form id="predictionForm">
                <div class="row">
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label for="radius_mean" class="form-label"
                        >Radius (mean)</label
                      >
                      <input
                        type="number"
                        class="form-control"
                        id="radius_mean"
                        step="0.01"
                        required
                      />
                    </div>
                    <div class="mb-3">
                      <label for="texture_mean" class="form-label"
                        >Texture (mean)</label
                      >
                      <input
                        type="number"
                        class="form-control"
                        id="texture_mean"
                        step="0.01"
                        required
                      />
                    </div>
                    <div class="mb-3">
                      <label for="perimeter_mean" class="form-label"
                        >Perimeter (mean)</label
                      >
                      <input
                        type="number"
                        class="form-control"
                        id="perimeter_mean"
                        step="0.01"
                        required
                      />
                    </div>
                  </div>
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label for="area_mean" class="form-label"
                        >Area (mean)</label
                      >
                      <input
                        type="number"
                        class="form-control"
                        id="area_mean"
                        step="0.01"
                        required
                      />
                    </div>
                    <div class="mb-3">
                      <label for="smoothness_mean" class="form-label"
                        >Smoothness (mean)</label
                      >
                      <input
                        type="number"
                        class="form-control"
                        id="smoothness_mean"
                        step="0.01"
                        required
                      />
                    </div>
                    <div class="mb-3">
                      <label for="compactness_mean" class="form-label"
                        >Compactness (mean)</label
                      >
                      <input
                        type="number"
                        class="form-control"
                        id="compactness_mean"
                        step="0.01"
                        required
                      />
                    </div>
                  </div>
                </div>

                <div class="text-center">
                  <button type="submit" class="btn btn-primary btn-lg">
                    🔍 Analyze Sample
                  </button>
                </div>
              </form>
            </div>
          </div>

          <!-- Results Section -->
          <div id="results" class="result-card" style="display: none;">
            <div class="card" id="resultCard">
              <div class="card-body">
                <h5 class="card-title">📊 Analysis Results</h5>
                <div class="row">
                  <div class="col-md-6">
                    <h6>Prediction:</h6>
                    <h4 id="prediction" class="mb-3"></h4>
                  </div>
                  <div class="col-md-6">
                    <h6>Confidence:</h6>
                    <h4 id="confidence" class="mb-3"></h4>
                  </div>
                </div>
                <div class="progress mb-3">
                  <div
                    id="confidenceBar"
                    class="progress-bar"
                    role="progressbar"
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Medical Disclaimer -->
          <div class="alert disclaimer mt-4" role="alert">
            <h6><strong>⚕️ Medical Disclaimer:</strong></h6>
            This tool is for educational and research purposes only. It should
            not be used as a substitute for professional medical advice,
            diagnosis, or treatment. Always consult with qualified healthcare
            professionals.
          </div>
        </div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
      document
        .getElementById("predictionForm")
        .addEventListener("submit", async function (e) {
          e.preventDefault();

          // Collect form data (simplified - would need all 30 features)
          const features = [
            parseFloat(document.getElementById("radius_mean").value),
            parseFloat(document.getElementById("texture_mean").value),
            parseFloat(document.getElementById("perimeter_mean").value),
            parseFloat(document.getElementById("area_mean").value),
            parseFloat(document.getElementById("smoothness_mean").value),
            parseFloat(document.getElementById("compactness_mean").value),
            // Add remaining 24 features with default values for demo
            ...Array(24).fill(0.1),
          ];

          try {
            const response = await fetch("/predict", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ features: features }),
            });

            const result = await response.json();

            if (result.error) {
              alert("Error: " + result.error);
              return;
            }

            // Display results
            document.getElementById("prediction").textContent =
              result.prediction;
            document.getElementById("confidence").textContent =
              (result.confidence * 100).toFixed(1) + "%";

            const resultCard = document.getElementById("resultCard");
            const confidenceBar = document.getElementById("confidenceBar");

            if (result.prediction === "Malignant") {
              resultCard.classList.add("malignant");
              resultCard.classList.remove("benign");
              confidenceBar.className = "progress-bar bg-danger";
            } else {
              resultCard.classList.add("benign");
              resultCard.classList.remove("malignant");
              confidenceBar.className = "progress-bar bg-success";
            }

            confidenceBar.style.width = result.confidence * 100 + "%";
            document.getElementById("results").style.display = "block";
          } catch (error) {
            alert("Error making prediction: " + error.message);
          }
        });
    </script>
  </body>
</html>
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs plots

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  breast-cancer-app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - TF_CPP_MIN_LOG_LEVEL=2
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - breast-cancer-app
    restart: unless-stopped

volumes:
  model_data:
  logs:
```

### Building and Running

```bash
# Build the Docker image
docker build -t breast-cancer-classifier .

# Run the container
docker run -p 5000:5000 breast-cancer-classifier

# Using Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f breast-cancer-app

# Scale the application
docker-compose up -d --scale breast-cancer-app=3
```

## ☁️ Cloud Deployment

### AWS Deployment with Elastic Beanstalk

```python
# application.py (AWS Elastic Beanstalk)
from flask import Flask
import os

# Import your main application
from app import app as application

if __name__ == "__main__":
    application.run(debug=False, port=int(os.environ.get('PORT', 5000)))
```

### Requirements for AWS

```txt
# requirements-aws.txt
Flask==2.3.2
tensorflow==2.13.0
scikit-learn==1.3.0
numpy==1.24.3
joblib==1.3.1
gunicorn==20.1.0
```

### Deployment Configuration

```yaml
# .ebextensions/python.config
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: application.py
  aws:elasticbeanstalk:application:environment:
    PYTHONPATH: "/var/app/current:$PYTHONPATH"
  aws:autoscaling:launchconfiguration:
    InstanceType: t3.medium
    IamInstanceProfile: aws-elasticbeanstalk-ec2-role
```

### Google Cloud Platform Deployment

```yaml
# app.yaml (Google App Engine)
runtime: python39

env_variables:
  TF_CPP_MIN_LOG_LEVEL: 2

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6

resources:
  cpu: 2
  memory_gb: 4
  disk_size_gb: 10

handlers:
  - url: /static
    static_dir: static
  - url: /.*
    script: auto
```

## 📊 Model Serving with TensorFlow Serving

### Model Export for TensorFlow Serving

```python
# export_model.py
import tensorflow as tf
import os

def export_model_for_serving():
    """Export model in TensorFlow Serving format."""

    # Load the trained model
    model = tf.keras.models.load_model('models/saved_models/breast_cancer_model.h5')

    # Export path
    export_path = 'models/tf_serving/breast_cancer_classifier/1'

    # Export the model
    tf.saved_model.save(model, export_path)

    print(f"Model exported to {export_path}")

    # Create model config
    model_config = {
        "model_config_list": {
            "config": [{
                "name": "breast_cancer_classifier",
                "base_path": "/models/breast_cancer_classifier",
                "model_platform": "tensorflow"
            }]
        }
    }

    return model_config

if __name__ == "__main__":
    export_model_for_serving()
```

### TensorFlow Serving Docker Setup

```bash
# Pull TensorFlow Serving image
docker pull tensorflow/serving

# Run TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/models/tf_serving,target=/models \
  -e MODEL_NAME=breast_cancer_classifier \
  -t tensorflow/serving
```

### Client Code for TensorFlow Serving

```python
# serving_client.py
import requests
import json
import numpy as np

def predict_with_tf_serving(features, server_url="http://localhost:8501"):
    """Make predictions using TensorFlow Serving."""

    # Prepare the request
    url = f"{server_url}/v1/models/breast_cancer_classifier:predict"

    # Format input data
    data = {
        "instances": features.tolist()
    }

    # Make request
    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        predictions = np.array(result['predictions'])
        return predictions
    else:
        raise Exception(f"Prediction failed: {response.text}")

# Example usage
if __name__ == "__main__":
    # Sample features (scaled)
    sample_features = np.random.normal(0, 1, (1, 30))

    try:
        predictions = predict_with_tf_serving(sample_features)
        print(f"Prediction: {predictions[0][0]:.4f}")
        print(f"Classification: {'Malignant' if predictions[0][0] > 0.5 else 'Benign'}")
    except Exception as e:
        print(f"Error: {e}")
```

## 📈 Monitoring and Maintenance

### Application Monitoring

```python
# monitoring.py
import logging
import time
from functools import wraps
from prometheus_client import Counter, Histogram, generate_latest
import psutil

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions made', ['result'])
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction processing time')
model_accuracy = Histogram('model_accuracy', 'Model accuracy over time')

def monitor_prediction(func):
    """Decorator to monitor prediction performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            prediction_counter.labels(result='success').inc()
            return result
        except Exception as e:
            prediction_counter.labels(result='error').inc()
            raise
        finally:
            duration = time.time() - start_time
            prediction_duration.observe(duration)

    return wrapper

def get_system_metrics():
    """Get system performance metrics."""
    return {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'model_loaded': model is not None
    }

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()
```

### Health Monitoring

```python
# health_check.py
import tensorflow as tf
import numpy as np
import time
from datetime import datetime

class HealthChecker:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.last_check = datetime.now()

    def comprehensive_health_check(self):
        """Perform comprehensive health check."""
        checks = {
            'model_loaded': self._check_model_loaded(),
            'scaler_loaded': self._check_scaler_loaded(),
            'prediction_test': self._test_prediction(),
            'memory_usage': self._check_memory_usage(),
            'response_time': self._check_response_time()
        }

        all_healthy = all(checks.values())

        return {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'checks': checks
        }

    def _check_model_loaded(self):
        """Check if model is properly loaded."""
        return self.model is not None

    def _check_scaler_loaded(self):
        """Check if scaler is properly loaded."""
        return self.scaler is not None

    def _test_prediction(self):
        """Test prediction with dummy data."""
        try:
            dummy_data = np.random.normal(0, 1, (1, 30))
            dummy_scaled = self.scaler.transform(dummy_data)
            prediction = self.model.predict(dummy_scaled, verbose=0)
            return 0 <= prediction[0][0] <= 1
        except Exception:
            return False

    def _check_memory_usage(self):
        """Check memory usage."""
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 85  # Threshold: 85%

    def _check_response_time(self):
        """Check prediction response time."""
        try:
            start_time = time.time()
            dummy_data = np.random.normal(0, 1, (1, 30))
            dummy_scaled = self.scaler.transform(dummy_data)
            self.model.predict(dummy_scaled, verbose=0)
            response_time = time.time() - start_time
            return response_time < 1.0  # Threshold: 1 second
        except Exception:
            return False
```

## 🔒 Security Considerations

### Input Validation

```python
# security.py
from flask import request
import numpy as np
from functools import wraps

def validate_input(f):
    """Decorator to validate input data."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()

        # Check if data exists
        if not data or 'features' not in data:
            return {'error': 'Invalid input: no features provided'}, 400

        features = data['features']

        # Validate feature count
        if len(features) != 30:
            return {'error': f'Invalid input: expected 30 features, got {len(features)}'}, 400

        # Validate feature types and ranges
        try:
            features_array = np.array(features, dtype=float)

            # Check for invalid values
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                return {'error': 'Invalid input: features contain NaN or infinite values'}, 400

            # Basic range validation (adjust based on your data)
            if np.any(features_array < 0) or np.any(features_array > 1000):
                return {'error': 'Invalid input: features out of expected range'}, 400

        except (ValueError, TypeError):
            return {'error': 'Invalid input: features must be numeric'}, 400

        return f(*args, **kwargs)

    return decorated_function

def rate_limit_by_ip():
    """Simple rate limiting by IP address."""
    # Implementation would depend on your chosen rate limiting library
    pass

def sanitize_output(prediction_result):
    """Sanitize output to prevent information leakage."""
    return {
        'prediction': prediction_result.get('prediction'),
        'confidence': round(prediction_result.get('confidence', 0), 3),
        'risk_level': prediction_result.get('risk_level'),
        'timestamp': prediction_result.get('timestamp')
    }
```

### HTTPS Configuration

```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://breast-cancer-app:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🚀 Deployment Checklist

### Pre-deployment

- [ ] Model training completed and validated
- [ ] Model exported in correct format
- [ ] Feature scaler saved and tested
- [ ] All dependencies documented in requirements.txt
- [ ] Environment variables configured
- [ ] Security measures implemented
- [ ] Health checks implemented
- [ ] Monitoring setup configured

### Deployment

- [ ] Application deployed to target environment
- [ ] Health checks passing
- [ ] SSL certificates configured (for production)
- [ ] Load balancer configured (if needed)
- [ ] Monitoring dashboards setup
- [ ] Backup procedures in place
- [ ] Rollback plan prepared

### Post-deployment

- [ ] Smoke tests completed
- [ ] Performance benchmarks met
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team trained on maintenance procedures
- [ ] Incident response plan activated

---

_This deployment guide provides comprehensive strategies for deploying the breast cancer classification model across various environments and platforms._
