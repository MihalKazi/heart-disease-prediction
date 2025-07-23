# Heart Disease Prediction Configuration

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
PAGE_TITLE = 'Heart Disease Prediction'
PAGE_ICON = ':heart:'

# Model Names
MODEL_NAMES = {
    'LogisticRegression': 'Logistic Regression',
    'RandomForestClassifier': 'Random Forest',
    'KNeighborsClassifier': 'K-Nearest Neighbors'
}
