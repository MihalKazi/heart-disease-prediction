"""
Project Setup Script
Run this script to set up the complete project structure and initial files.
"""

import os
import pandas as pd
import numpy as np

def create_project_structure():
    """Create the complete project directory structure"""
    directories = [
        'data',
        'models', 
        'logs',
        'visualizations',
        'reports',
        'tests'
    ]
    
    print("Creating project directory structure...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}/")
        else:
            print(f"✓ Directory already exists: {directory}/")

def create_sample_dataset():
    """Create a sample heart disease dataset if it doesn't exist"""
    data_path = 'data/heart_disease.csv'
    
    if not os.path.exists(data_path):
        print("\nCreating sample heart disease dataset...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic heart disease data
        data = {
            'age': np.clip(np.random.normal(55, 12, n_samples), 20, 100).astype(int),
            'sex': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'trestbps': np.clip(np.random.normal(130, 20, n_samples), 80, 200),
            'chol': np.clip(np.random.normal(240, 50, n_samples), 100, 600),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
            'thalach': np.clip(np.random.normal(150, 25, n_samples), 60, 220),
            'exang': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'oldpeak': np.clip(np.random.exponential(1, n_samples), 0, 6),
            'slope': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            'ca': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
            'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.1, 0.15, 0.05])
        }
        
        # Create target variable with realistic correlations
        risk_factors = (
            0.3 * (data['age'] > 60) +
            0.2 * data['sex'] +
            0.25 * (data['cp'] > 1) +
            0.2 * (data['chol'] > 250) +
            0.15 * data['exang'] +
            0.15 * (data['thalach'] < 120) +
            0.1 * (data['trestbps'] > 140) +
            0.1 * (data['oldpeak'] > 2) +
            0.1 * (data['ca'] > 0)
        )
        
        # Normalize and create binary target
        risk_prob = np.clip(risk_factors / 3, 0.1, 0.9)
        data['target'] = np.random.binomial(1, risk_prob, n_samples)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        print(f"✓ Sample dataset created and saved to {data_path}")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Features: {len(df.columns) - 1}")
        print(f"  - Target distribution: {df['target'].value_counts().to_dict()}")
    else:
        print(f"✓ Dataset already exists: {data_path}")

def create_gitignore():
    """Create a .gitignore file for the project"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environments
.env
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
logs/*.csv
models/*.pkl
visualizations/*.png
reports/*.pdf

# But keep directory structure
!logs/.gitkeep
!models/.gitkeep
!visualizations/.gitkeep
!reports/.gitkeep
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore file")

def create_gitkeep_files():
    """Create .gitkeep files to maintain directory structure in git"""
    directories = ['logs', 'models', 'visualizations', 'reports']
    
    for directory in directories:
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('')
            print(f"✓ Created {gitkeep_path}")

def create_config_file():
    """Create a configuration file for the project"""
    config_content = """# Heart Disease Prediction Configuration

# Model Settings
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature Columns
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

TARGET_COLUMN = 'target'

# File Paths
DATA_PATH = 'data/heart_disease.csv'
MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'
LOG_PATH = 'logs/prediction_log.csv'

# Streamlit Settings
PAGE_TITLE = "Heart Disease Prediction"
PAGE_ICON = "❤️"

# Model Names
MODEL_NAMES = {
    'LogisticRegression': 'Logistic Regression',
    'RandomForestClassifier': 'Random Forest',
    'KNeighborsClassifier': 'K-Nearest Neighbors'
}
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("✓ Created config.py")

def create_test_file():
    """Create basic test file"""
    test_content = """import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestHeartDiseaseModel:
    
    def test_data_loading(self):
        \"\"\"Test if data can be loaded successfully\"\"\"
        assert os.path.exists('data/heart_disease.csv'), "Dataset file not found"
        df = pd.read_csv('data/heart_disease.csv')
        assert len(df) > 0, "Dataset is empty"
        assert 'target' in df.columns, "Target column not found"
    
    def test_model_files_exist(self):
        \"\"\"Test if model files exist after training\"\"\"
        if os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl'):
            model = joblib.load('models/model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            assert model is not None, "Model is None"
            assert scaler is not None, "Scaler is None"
    
    def test_prediction_format(self):
        \"\"\"Test prediction output format\"\"\"
        if os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl'):
            model = joblib.load('models/model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            
            # Create sample input
            sample_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
            sample_scaled = scaler.transform(sample_data)
            
            prediction = model.predict(sample_scaled)
            probability = model.predict_proba(sample_scaled)
            
            assert len(prediction) == 1, "Prediction should return single value"
            assert prediction[0] in [0, 1], "Prediction should be 0 or 1"
            assert len(probability[0]) == 2, "Probability should have 2 values"
            assert abs(sum(probability[0]) - 1.0) < 0.001, "Probabilities should sum to 1"

if __name__ == "__main__":
    pytest.main([__file__])
"""
    
    test_dir = 'tests'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    with open(os.path.join(test_dir, 'test_model.py'), 'w') as f:
        f.write(test_content)
    print("✓ Created tests/test_model.py")

def main():
    """Main setup function"""
    print("🏗️  Setting up Heart Disease Prediction Project")
    print("=" * 50)
    
    # Create directory structure
    create_project_structure()
    
    # Create sample dataset
    create_sample_dataset()
    
    # Create configuration files
    create_gitignore()
    create_gitkeep_files()
    create_config_file()
    create_test_file()
    
    print("\n" + "=" * 50)
    print("✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train the model: python model_training.py")
    print("3. Run the app: streamlit run app.py")
    print("4. Run tests: python -m pytest tests/")
    print("\n📖 Check README.md for detailed documentation")

if __name__ == "__main__":
    main()