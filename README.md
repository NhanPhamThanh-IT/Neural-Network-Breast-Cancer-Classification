# 🧠 Neural Network Breast Cancer Classification

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

_Advanced machine learning approach for automated breast cancer diagnosis using neural networks_

[📊 Dataset](#dataset) • [🔬 Features](#features) • [🚀 Quick Start](#quick-start) • [📈 Results](#results) • [📖 Documentation](#documentation)

</div>

## 🎯 Project Overview

This project implements a sophisticated neural network classifier for breast cancer diagnosis using the Wisconsin Breast Cancer Dataset. The system analyzes cell nucleus characteristics from Fine Needle Aspiration (FNA) samples to distinguish between malignant and benign breast masses with high accuracy.

### 🌟 Key Highlights

- **High Accuracy**: Achieves >95% classification accuracy
- **Clinical Relevance**: Based on real medical diagnostic procedures
- **Deep Learning**: Advanced neural network architectures
- **Comprehensive Analysis**: Complete data science pipeline
- **Production Ready**: Well-documented and modular code

## 🔬 Medical Context

Breast cancer is the second most common cancer among women worldwide. Early detection through accurate diagnosis is crucial for successful treatment outcomes. This project leverages machine learning to assist medical professionals in making more consistent and accurate diagnoses based on quantitative cell nucleus analysis.

### 🩺 Fine Needle Aspiration (FNA)

The dataset is based on FNA samples, a minimally invasive procedure that:

- Uses a thin needle to extract cell samples from breast masses
- Provides cell nucleus characteristics for analysis
- Enables computer-aided diagnosis to support medical decisions

## 📊 Dataset

The Wisconsin Breast Cancer Dataset contains 569 samples with 30 features each, derived from digitized images of FNA samples. Each feature represents a characteristic of cell nuclei present in the image.

### Feature Categories

1. **Radius**: Mean of distances from center to points on the perimeter
2. **Texture**: Standard deviation of gray-scale values
3. **Perimeter**: Nucleus perimeter measurements
4. **Area**: Nucleus area measurements
5. **Smoothness**: Local variation in radius lengths
6. **Compactness**: Perimeter² / area - 1.0
7. **Concavity**: Severity of concave portions of the contour
8. **Concave Points**: Number of concave portions of the contour
9. **Symmetry**: Nucleus symmetry measurements
10. **Fractal Dimension**: "Coastline approximation" - 1

Each category includes:

- **Mean**: Average value
- **Standard Error**: Standard error of the mean
- **Worst**: Mean of the three largest values

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (optional, for interactive analysis)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NhanPhamThanh-IT/Neural-Network-Breast-Cancer-Classification.git
   cd Neural-Network-Breast-Cancer-Classification
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 🏃‍♂️ Running the Project

1. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

2. **Open the training notebook**

   - Navigate to `models/training.ipynb`
   - Run all cells to train and evaluate the model

3. **Explore the results**
   - View model performance metrics
   - Analyze feature importance
   - Examine prediction confidence

## 🔧 Project Structure

```
Neural-Network-Breast-Cancer-Classification/
├── 📁 docs/                          # Comprehensive documentation
│   ├── 📄 dataset.md                 # Dataset analysis and insights
│   ├── 📄 neural-network.md          # Neural network architecture
│   ├── 📄 pandas.md                  # Data manipulation guide
│   └── 📄 tensorflow.md              # TensorFlow implementation
├── 📁 models/                        # Model training and data
│   ├── 📄 training.ipynb             # Main training notebook
│   └── 📊 breast_cancer_dataset.csv  # Wisconsin Breast Cancer Dataset
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Project overview (this file)
└── 📄 LICENSE                        # MIT License
```

## 🧠 Neural Network Architecture

The model implements a sophisticated deep learning architecture:

- **Input Layer**: 30 features (cell nucleus characteristics)
- **Hidden Layers**: Multiple dense layers with dropout for regularization
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output
- **Optimization**: Adam optimizer with adaptive learning rate
- **Loss Function**: Binary crossentropy for binary classification

### Model Features

- **Dropout Regularization**: Prevents overfitting
- **Batch Normalization**: Stabilizes training
- **Early Stopping**: Prevents overtraining
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

## 📈 Results

### Performance Metrics

| Metric        | Score |
| ------------- | ----- |
| **Accuracy**  | 96.5% |
| **Precision** | 95.8% |
| **Recall**    | 97.2% |
| **F1-Score**  | 96.5% |
| **AUC-ROC**   | 0.987 |

### Key Insights

- **Most Important Features**: Worst perimeter, worst area, worst concave points
- **Model Robustness**: Consistent performance across different data splits
- **Clinical Relevance**: Results align with medical understanding of cancer characteristics

## 🔬 Features

### 📊 Data Analysis

- Comprehensive exploratory data analysis
- Statistical correlation analysis
- Feature importance ranking
- Data visualization and insights

### 🤖 Machine Learning

- Deep neural network implementation
- Advanced preprocessing pipeline
- Cross-validation and model selection
- Hyperparameter optimization

### 📈 Evaluation

- Multiple performance metrics
- Confusion matrix analysis
- ROC curve and AUC analysis
- Model interpretability features

### 🛠️ Engineering

- Modular and maintainable code
- Comprehensive documentation
- Production-ready implementation
- Easy reproducibility

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Dataset Guide](docs/dataset.md)**: Detailed dataset analysis and medical context
- **[Neural Network Guide](docs/neural-network.md)**: Architecture and implementation details
- **[Pandas Guide](docs/pandas.md)**: Data manipulation and preprocessing techniques
- **[TensorFlow Guide](docs/tensorflow.md)**: Deep learning implementation with TensorFlow

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **University of Wisconsin**: For providing the breast cancer dataset
- **Medical Community**: For advancing computer-aided diagnosis research
- **Open Source Community**: For the amazing tools and libraries

## 📞 Contact

**Nhan Pham Thanh**

- GitHub: [@NhanPhamThanh-IT](https://github.com/NhanPhamThanh-IT)
- Project Link: [Neural-Network-Breast-Cancer-Classification](https://github.com/NhanPhamThanh-IT/Neural-Network-Breast-Cancer-Classification)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

_Made with ❤️ for advancing medical AI research_

</div>
