heart-disease-prediction/
├── 📄 Core Application Files
│   ├── app.py                     # Main Streamlit web application
│   ├── model_training.py          # ML model training script
│   ├── batch_predict.py           # Standalone batch prediction tool
│   └── setup.py                   # Project setup and initialization
│
├── ⚙️ Configuration & Deployment
│   ├── config.py                  # Project configuration settings
│   ├── requirements.txt           # Python dependencies
│   ├── Dockerfile                 # Docker container configuration
│   ├── docker-compose.yml         # Docker Compose setup
│   ├── deploy.sh                  # Automated deployment script
│   └── Makefile                   # Build and management commands
│
├── 📊 Data & Models
│   ├── data/
│   │   ├── heart_disease.csv      # Main dataset (auto-generated)
│   │   ├── sample_batch.csv       # Sample batch prediction file
│   │   └── .gitkeep              # Keep directory in git
│   ├── models/
│   │   ├── model.pkl              # Trained ML model
│   │   ├── scaler.pkl             # Feature scaler
│   │   ├── model_info.txt         # Model metadata
│   │   └── .gitkeep              # Keep directory in git
│
├── 📈 Outputs & Logs
│   ├── logs/
│   │   ├── prediction_log.csv     # Individual prediction history
│   │   ├── batch_prediction_log.csv # Batch prediction logs
│   │   ├── training_log.txt       # Training process logs
│   │   └── .gitkeep              # Keep directory in git
│   ├── visualizations/
│   │   ├── eda_plots.png          # Exploratory data analysis
│   │   ├── confusion_matrices/    # Model performance plots
│   │   ├── feature_importance.png # Feature importance charts
│   │   └── .gitkeep              # Keep directory in git
│   └── reports/
│       ├── *.pdf                  # Generated prediction reports
│       └── .gitkeep              # Keep directory in git
│
├── 🧪 Testing & Documentation
│   ├── tests/
│   │   ├── test_model.py          # Model testing suite
│   │   └── __init__.py
│   ├── README.md                  # Detailed project documentation
│   ├── PROJECT_OVERVIEW.md        # This file
│   └── .gitignore                 # Git ignore rules
│
└── 🔧 Development Tools
    ├── .github/                   # GitHub Actions (optional)
    ├── docs/                      # Additional documentation
    └── scripts/                   # Utility scripts