#!/usr/bin/env python3
"""
Setup script for Neural Network Breast Cancer Classification project.

This script helps users set up the project environment and verify installation.
"""

import sys
import subprocess
import platform
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_installations():
    """Verify that all required packages are installed correctly."""
    required_packages = [
        "numpy",
        "pandas", 
        "scikit-learn",
        "tensorflow",
        "jupyter",
        "ipykernel"
    ]
    
    print("\n🔍 Verifying package installations...")
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: Not installed")
            return False
    
    return True

def test_imports():
    """Test importing key packages."""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", None),
        ("tensorflow", "tf"),
        ("jupyter", None)
    ]
    
    for package, alias in test_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
                print(f"✅ {package} imported successfully")
            else:
                exec(f"import {package}")
                print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            return False
    
    return True

def check_tensorflow_gpu():
    """Check if TensorFlow can access GPU."""
    try:
        import tensorflow as tf
        
        print(f"\n🤖 TensorFlow version: {tf.__version__}")
        
        # Check for GPU availability
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"✅ GPU devices found: {len(physical_devices)}")
            for i, device in enumerate(physical_devices):
                print(f"   GPU {i}: {device}")
        else:
            print("ℹ️  No GPU devices found - using CPU")
            print("   For GPU support, install tensorflow-gpu and CUDA/cuDNN")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow GPU check failed: {e}")
        return False

def setup_jupyter():
    """Set up Jupyter kernel for the project."""
    try:
        print("\n📓 Setting up Jupyter kernel...")
        subprocess.check_call([
            sys.executable, "-m", "ipykernel", "install", "--user", 
            "--name", "breast-cancer-nn", "--display-name", "Breast Cancer NN"
        ])
        print("✅ Jupyter kernel installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup Jupyter kernel: {e}")
        return False

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "logs",
        "models/saved_models",
        "models/checkpoints",
        "outputs",
        "plots"
    ]
    
    print("\n📁 Creating project directories...")
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def run_basic_test():
    """Run a basic test to ensure everything is working."""
    print("\n🔬 Running basic functionality test...")
    
    try:
        # Test data loading
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        print("✅ Data manipulation test passed")
        
        # Test TensorFlow
        import tensorflow as tf
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy')
        print("✅ TensorFlow model creation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Launch Jupyter Notebook:")
    print("   jupyter notebook")
    print("\n2. Open the training notebook:")
    print("   models/training.ipynb")
    print("\n3. Start exploring the project:")
    print("   - Read the documentation in docs/")
    print("   - Check out the README.md for overview")
    print("   - Run the training notebook cells")
    print("\n4. For development:")
    print("   - Read CONTRIBUTING.md for guidelines")
    print("   - Check CHANGELOG.md for version history")
    print("\n💡 Tips:")
    print("   - Use 'Breast Cancer NN' kernel in Jupyter for this project")
    print("   - Check GPU availability with TensorFlow if needed")
    print("   - Visit docs/ folder for detailed guides")
    print("\n🆘 Need help? Check the documentation or create an issue!")

def main():
    """Main setup function."""
    print("🚀 Neural Network Breast Cancer Classification - Setup")
    print("="*60)
    
    # System information
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    
    # Run setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing requirements", install_requirements),
        ("Verifying installations", verify_installations),
        ("Testing imports", test_imports),
        ("Checking TensorFlow/GPU", check_tensorflow_gpu),
        ("Setting up Jupyter", setup_jupyter),
        ("Creating directories", create_directories),
        ("Running basic test", run_basic_test)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please check the error messages above and try again.")
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()
