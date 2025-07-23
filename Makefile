# Heart Disease Prediction Project Makefile
# Provides convenient commands for project management

.PHONY: help install setup train test run clean docker docker-build docker-run lint format check-deps

# Default target
help:
	@echo "Heart Disease Prediction Project"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  help          Show this help message"
	@echo "  install       Install dependencies"
	@echo "  setup         Set up project structure"
	@echo "  train         Train the machine learning model"
	@echo "  test          Run tests"
	@echo "  run           Start the Streamlit application"
	@echo "  clean         Clean generated files"
	@echo "  lint          Run code linting"
	@echo "  format        Format code with black"
	@echo "  check-deps    Check for dependency updates"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run with Docker"
	@echo "  docker        Build and run with Docker"
	@echo "  all           Complete setup (install + setup + train)"
	@echo ""

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Set up project structure
setup:
	@echo "Setting up project structure..."
	python setup.py
	@echo "✓ Project setup complete"

# Train the model
train:
	@echo "Training machine learning model..."
	python model_training.py
	@echo "✓ Model training complete"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --tb=short
	@echo "✓ Tests complete"

# Start the Streamlit application
run:
	@echo "Starting Streamlit application..."
	@echo "Visit http://localhost:8501 in your browser"
	streamlit run app.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf models/*.pkl
	rm -rf logs/*.csv
	rm -rf visualizations/*.png
	rm -rf reports/*.pdf
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "✓ Cleanup complete"

# Lint code
lint:
	@echo "Running code linting..."
	flake8 --max-line-length=88 --extend-ignore=E203,W503 *.py
	@echo "✓ Linting complete"

# Format code
format:
	@echo "Formatting code with black..."
	black --line-length=88 *.py
	@echo "✓ Code formatting complete"

# Check for dependency updates
check-deps:
	@echo "Checking for dependency updates..."
	pip list --outdated
	@echo "✓ Dependency check complete"

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t heart-disease-prediction .
	@echo "✓ Docker image built"

# Run with Docker
docker-run:
	@echo "Running with Docker..."
	docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models heart-disease-prediction

# Build and run with Docker Compose
docker:
	@echo "Building and running with Docker Compose..."
	docker-compose up --build

# Complete setup
all: install setup train
	@echo "✅ Complete setup finished!"
	@echo "Run 'make run' to start the application"

# Development setup
dev-install:
	@echo "Installing development dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black flake8 pytest pytest-cov
	@echo "✓ Development dependencies installed"

# Run with coverage
test-coverage:
	@echo "Running tests with coverage..."
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

# Quick development cycle
dev: format lint test
	@echo "✓ Development cycle complete"

# Create sample batch file for testing
create-sample:
	@echo "Creating sample batch prediction file..."
	python -c "
import pandas as pd
import numpy as np
np.random.seed(42)
data = {
    'age': np.random.randint(30, 80, 10),
    'sex': np.random.choice([0, 1], 10),
    'cp': np.random.choice([0, 1, 2, 3], 10),
    'trestbps': np.random.randint(90, 180, 10),
    'chol': np.random.randint(150, 400, 10),
    'fbs': np.random.choice([0, 1], 10),
    'restecg': np.random.choice([0, 1, 2], 10),
    'thalach': np.random.randint(100, 200, 10),
    'exang': np.random.choice([0, 1], 10),
    'oldpeak': np.random.uniform(0, 4, 10).round(1),
    'slope': np.random.choice([0, 1, 2], 10),
    'ca': np.random.choice([0, 1, 2, 3], 10),
    'thal': np.random.choice([0, 1, 2, 3], 10)
}
pd.DataFrame(data).to_csv('data/sample_batch.csv', index=False)
print('Sample batch file created: data/sample_batch.csv')
"
	@echo "✓ Sample batch file created"

# Batch prediction example
batch-predict: create-sample
	@echo "Running batch prediction example..."
	python batch_predict.py data/sample_batch.csv data/sample_results.csv --summary --log
	@echo "✓ Batch prediction example complete"

# Performance benchmark
benchmark:
	@echo "Running performance benchmark..."
	python -c "
import time
import pandas as pd
import joblib
import numpy as np
from model_training import load_data
print('Loading model and data...')
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
df = pd.read_csv('data/heart_disease.csv')
X = df.drop('target', axis=1)
X_scaled = scaler.transform(X)
print(f'Benchmarking with {len(X)} samples...')
start_time = time.time()
predictions = model.predict(X_scaled)
end_time = time.time()
print(f'Prediction time: {end_time - start_time:.4f} seconds')
print(f'Predictions per second: {len(X) / (end_time - start_time):.0f}')
"
	@echo "✓ Benchmark complete"