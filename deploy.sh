#!/bin/bash

# Heart Disease Prediction Project Deployment Script
# This script automates the setup and deployment process

set -e  # Exit on any error

echo "🚀 Heart Disease Prediction Project Deployment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION is installed"
    else
        print_error "Python 3 is not installed. Please install Python 3.7 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    if command -v pip3 &> /dev/null; then
        print_status "pip3 is available"
    else
        print_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_status "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Dependencies installed successfully"
}

# Run setup script
run_setup() {
    print_info "Running project setup..."
    python setup.py
    print_status "Project setup completed"
}

# Train the model
train_model() {
    print_info "Training machine learning model..."
    python model_training.py
    print_status "Model training completed"
}

# Test the installation
test_installation() {
    print_info "Testing the installation..."
    if [ -f "models/model.pkl" ] && [ -f "models/scaler.pkl" ]; then
        print_status "Model files found"
    else
        print_warning "Model files not found. Training may have failed."
    fi
    
    if [ -f "data/heart_disease.csv" ]; then
        print_status "Dataset found"
    else
        print_warning "Dataset not found"
    fi
}

# Function to start the application
start_app() {
    print_info "Starting Streamlit application..."
    echo ""
    echo "🌐 The application will be available at: http://localhost:8501"
    echo "📊 Use Ctrl+C to stop the application"
    echo ""
    streamlit run app.py
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --full          Full deployment (setup + train + run)"
    echo "  --setup-only    Only run setup and install dependencies"
    echo "  --train-only    Only train the model"
    echo "  --run-only      Only start the application"
    echo "  --test          Run tests"
    echo "  --clean         Clean up generated files"
    echo "  --docker        Deploy using Docker"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --full       # Complete setup and deployment"
    echo "  $0 --run-only   # Just start the app (if already set up)"
    echo "  $0 --docker     # Deploy using Docker"
}

# Function to clean up
clean_up() {
    print_info "Cleaning up generated files..."
    
    # Remove model files
    if [ -d "models" ]; then
        rm -f models/*.pkl models/*.txt
        print_status "Removed model files"
    fi
    
    # Remove logs
    if [ -d "logs" ]; then
        rm -f logs/*.csv
        print_status "Removed log files"
    fi
    
    # Remove visualizations
    if [ -d "visualizations" ]; then
        rm -f visualizations/*.png
        print_status "Removed visualization files"
    fi
    
    # Remove reports
    if [ -d "reports" ]; then
        rm -f reports/*.pdf
        print_status "Removed report files"
    fi
    
    print_status "Cleanup completed"
}

# Function to run tests
run_tests() {
    print_info "Running tests..."
    if [ -f "tests/test_model.py" ]; then
        python -m pytest tests/ -v
        print_status "Tests completed"
    else
        print_warning "Test files not found"
    fi
}

# Function to deploy with Docker
deploy_docker() {
    print_info "Deploying with Docker..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi
    
    # Build and run with docker-compose
    print_info "Building Docker image..."
    docker-compose build
    
    print_info "Starting application with Docker..."
    echo ""
    echo "🌐 The application will be available at: http://localhost:8501"
    echo "📊 Use Ctrl+C to stop the application"
    echo ""
    docker-compose up
}

# Main deployment function
main_deployment() {
    print_info "Starting full deployment process..."
    
    check_python
    check_pip
    create_venv
    activate_venv
    install_dependencies
    run_setup
    train_model
    test_installation
    
    print_status "Deployment completed successfully!"
    echo ""
    echo "🎉 Your Heart Disease Prediction System is ready!"
    echo ""
    read -p "Do you want to start the application now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_app
    else
        echo ""
        echo "To start the application later, run:"
        echo "  source venv/bin/activate"
        echo "  streamlit run app.py"
    fi
}

# Parse command line arguments
case "$1" in
    --full)
        main_deployment
        ;;
    --setup-only)
        check_python
        check_pip
        create_venv
        activate_venv
        install_dependencies
        run_setup
        print_status "Setup completed!"
        ;;
    --train-only)
        activate_venv
        train_model
        ;;
    --run-only)
        if [ -d "venv" ]; then
            activate_venv
        fi
        start_app
        ;;
    --test)
        if [ -d "venv" ]; then
            activate_venv
        fi
        run_tests
        ;;
    --clean)
        clean_up
        ;;
    --docker)
        deploy_docker
        ;;
    --help)
        show_usage
        ;;
    "")
        # No arguments provided, show usage and run full deployment
        show_usage
        echo ""
        read -p "Do you want to run full deployment? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            main_deployment
        fi
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac