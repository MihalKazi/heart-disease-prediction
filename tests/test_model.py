import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestHeartDiseaseModel:
    
    def test_data_loading(self):
        """Test if data can be loaded successfully"""
        assert os.path.exists('data/heart_disease.csv'), "Dataset file not found"
        df = pd.read_csv('data/heart_disease.csv')
        assert len(df) > 0, "Dataset is empty"
        assert 'target' in df.columns, "Target column not found"
    
    def test_model_files_exist(self):
        """Test if model files exist after training"""
        if os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl'):
            model = joblib.load('models/model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            assert model is not None, "Model is None"
            assert scaler is not None, "Scaler is None"
    
    def test_prediction_format(self):
        """Test prediction output format"""
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
