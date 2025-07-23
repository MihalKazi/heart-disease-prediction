"""
Heart Disease Prediction Model Training Script
This script loads the heart disease dataset, preprocesses it, trains multiple models,
evaluates them, and saves the best performing model and scaler.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories for the project"""
    directories = ['data', 'models', 'logs', 'visualizations']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_data():
    """Load the heart disease dataset"""
    # If you have the UCI heart disease dataset, place it in the data folder
    # For this example, we'll create a sample dataset if the file doesn't exist
    
    data_path = 'data/heart_disease.csv'
    
    if not os.path.exists(data_path):
        print("Dataset not found. Creating sample dataset...")
        # Create sample data with realistic distributions
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.normal(55, 12, n_samples).astype(int),
            'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
            'cp': np.random.choice([0, 1, 2, 3], n_samples),  # Chest pain type
            'trestbps': np.random.normal(130, 20, n_samples),  # Resting blood pressure
            'chol': np.random.normal(240, 50, n_samples),  # Cholesterol
            'fbs': np.random.choice([0, 1], n_samples),  # Fasting blood sugar
            'restecg': np.random.choice([0, 1, 2], n_samples),  # Resting ECG
            'thalach': np.random.normal(150, 25, n_samples),  # Max heart rate
            'exang': np.random.choice([0, 1], n_samples),  # Exercise induced angina
            'oldpeak': np.random.exponential(1, n_samples),  # ST depression
            'slope': np.random.choice([0, 1, 2], n_samples),  # Slope of peak exercise ST
            'ca': np.random.choice([0, 1, 2, 3, 4], n_samples),  # Number of major vessels
            'thal': np.random.choice([0, 1, 2, 3], n_samples),  # Thalassemia
        }
        
        # Create target variable with some correlation to features
        heart_disease_prob = (
            0.3 * (data['age'] > 60) +
            0.2 * data['sex'] +
            0.25 * (data['cp'] > 1) +
            0.2 * (data['chol'] > 250) +
            0.15 * data['exang'] +
            0.1 * (data['thalach'] < 120)
        )
        
        data['target'] = np.random.binomial(1, np.clip(heart_disease_prob, 0, 1), n_samples)
        
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        print(f"Sample dataset created and saved to {data_path}")
    else:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded from {data_path}")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset info:")
    print(df.info())
    
    print("\nDataset description:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nTarget distribution:")
    print(df['target'].value_counts())
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Target distribution
    plt.subplot(2, 3, 1)
    df['target'].value_counts().plot(kind='bar')
    plt.title('Heart Disease Distribution')
    plt.xlabel('Heart Disease (0: No, 1: Yes)')
    plt.ylabel('Count')
    
    # Age distribution
    plt.subplot(2, 3, 2)
    plt.hist(df['age'], bins=20, alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Correlation heatmap
    plt.subplot(2, 3, 3)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    # Heart disease by gender
    plt.subplot(2, 3, 4)
    pd.crosstab(df['sex'], df['target']).plot(kind='bar')
    plt.title('Heart Disease by Gender')
    plt.xlabel('Gender (0: Female, 1: Male)')
    plt.ylabel('Count')
    plt.legend(['No Disease', 'Disease'])
    
    # Cholesterol distribution
    plt.subplot(2, 3, 5)
    plt.hist(df['chol'], bins=20, alpha=0.7)
    plt.title('Cholesterol Distribution')
    plt.xlabel('Cholesterol')
    plt.ylabel('Frequency')
    
    # Heart rate vs Age
    plt.subplot(2, 3, 6)
    plt.scatter(df['age'], df['thalach'], c=df['target'], alpha=0.6)
    plt.title('Max Heart Rate vs Age')
    plt.xlabel('Age')
    plt.ylabel('Max Heart Rate')
    plt.colorbar(label='Heart Disease')
    
    plt.tight_layout()
    plt.savefig('visualizations/eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def preprocess_data(df):
    """Preprocess the data for training"""
    print("\n=== DATA PREPROCESSING ===")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_models(X_train, y_train):
    """Train multiple models and return them"""
    print("\n=== MODEL TRAINING ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return the best one"""
    print("\n=== MODEL EVALUATION ===")
    
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'visualizations/{name.lower().replace(" ", "_")}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # Find best model based on F1-score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
    
    return best_model, best_model_name, results

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'{model_name} - Feature Importance')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.lower().replace(" ", "_")}_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def save_model_and_scaler(model, scaler, model_name):
    """Save the trained model and scaler"""
    model_path = 'models/model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n=== MODEL SAVED ===")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Save model info
    with open('models/model_info.txt', 'w') as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Features: {len(scaler.feature_names_in_)} features\n")

def main():
    """Main function to run the entire training pipeline"""
    print("=== HEART DISEASE PREDICTION MODEL TRAINING ===")
    
    # Create directories
    create_directories()
    
    # Load data
    df = load_data()
    
    # Explore data
    df = explore_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    best_model, best_model_name, results = evaluate_models(models, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(best_model, feature_names, best_model_name)
    
    # Save model and scaler
    save_model_and_scaler(best_model, scaler, best_model_name)
    
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()