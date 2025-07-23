"""
Standalone Batch Prediction Script for Heart Disease
This script allows command-line batch processing of heart disease predictions
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
from datetime import datetime
import sys

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run model_training.py first!")
        print(f"Missing file: {e}")
        sys.exit(1)

def validate_input_data(df):
    """Validate input data format and columns"""
    required_columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Required columns: {required_columns}")
        sys.exit(1)
    
    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        print("Warning: Missing values found in the following columns:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count} missing values")
        
        # Fill missing values with median for numeric columns
        for col in required_columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        print("Missing values filled with median/mode values.")
    
    return df[required_columns]

def make_predictions(model, scaler, df):
    """Make predictions on the input data"""
    try:
        # Scale the input data
        X_scaled = scaler.transform(df)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)

def create_output_dataframe(input_df, predictions, confidences):
    """Create output dataframe with predictions"""
    output_df = input_df.copy()
    output_df['prediction'] = predictions
    output_df['risk_level'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
    output_df['confidence'] = confidences
    output_df['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return output_df

def generate_summary_report(predictions, confidences):
    """Generate a summary report of the predictions"""
    total_patients = len(predictions)
    high_risk_count = sum(predictions)
    low_risk_count = total_patients - high_risk_count
    avg_confidence = np.mean(confidences)
    
    print("\n" + "="*50)
    print("BATCH PREDICTION SUMMARY REPORT")
    print("="*50)
    print(f"Total Patients Processed: {total_patients}")
    print(f"High Risk Predictions: {high_risk_count} ({high_risk_count/total_patients*100:.1f}%)")
    print(f"Low Risk Predictions: {low_risk_count} ({low_risk_count/total_patients*100:.1f}%)")
    print(f"Average Prediction Confidence: {avg_confidence:.2%}")
    print(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if high_risk_count > 0:
        high_risk_indices = np.where(predictions == 1)[0]
        high_risk_confidences = confidences[high_risk_indices]
        print(f"\nHigh Risk Patients Statistics:")
        print(f"  Average Confidence: {np.mean(high_risk_confidences):.2%}")
        print(f"  Min Confidence: {np.min(high_risk_confidences):.2%}")
        print(f"  Max Confidence: {np.max(high_risk_confidences):.2%}")
    
    print("="*50)

def save_detailed_log(output_df, filename):
    """Save detailed prediction log"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'batch_prediction_log_{filename}')
    output_df.to_csv(log_file, index=False)
    print(f"Detailed log saved to: {log_file}")

def main():
    """Main function for batch prediction"""
    parser = argparse.ArgumentParser(
        description='Heart Disease Batch Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python batch_predict.py input.csv output.csv
  python batch_predict.py data/patients.csv results/predictions.csv --summary
  python batch_predict.py input.csv output.csv --log --threshold 0.8
        '''
    )
    
    parser.add_argument('input_file', help='Input CSV file with patient data')
    parser.add_argument('output_file', help='Output CSV file for predictions')
    parser.add_argument('--summary', action='store_true', 
                       help='Display summary report')
    parser.add_argument('--log', action='store_true',
                       help='Save detailed log file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for high confidence predictions (default: 0.5)')
    parser.add_argument('--high-risk-only', action='store_true',
                       help='Only save high-risk predictions to output')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Heart Disease Batch Prediction Tool")
    print("="*40)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Confidence threshold: {args.threshold}")
    
    # Load model and scaler
    print("\nLoading trained model and scaler...")
    model, scaler = load_model_and_scaler()
    print("✓ Model and scaler loaded successfully")
    
    # Load input data
    print(f"\nLoading input data from {args.input_file}...")
    try:
        input_df = pd.read_csv(args.input_file)
        print(f"✓ Loaded {len(input_df)} patient records")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Validate input data
    print("\nValidating input data...")
    validated_df = validate_input_data(input_df)
    print("✓ Input data validated")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, confidences = make_predictions(model, scaler, validated_df)
    print("✓ Predictions completed")
    
    # Create output dataframe
    output_df = create_output_dataframe(validated_df, predictions, confidences)
    
    # Filter high-risk only if requested
    if args.high_risk_only:
        output_df = output_df[output_df['prediction'] == 1]
        print(f"✓ Filtered to {len(output_df)} high-risk patients only")
    
    # Filter by confidence threshold
    high_confidence_df = output_df[output_df['confidence'] >= args.threshold]
    if len(high_confidence_df) < len(output_df):
        print(f"Note: {len(high_confidence_df)} predictions meet confidence threshold of {args.threshold}")
    
    # Save output file
    try:
        output_df.to_csv(args.output_file, index=False)
        print(f"✓ Results saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    # Generate summary report if requested
    if args.summary:
        generate_summary_report(predictions, confidences)
    
    # Save detailed log if requested
    if args.log:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_detailed_log(output_df, f"{timestamp}.csv")
    
    # Show high confidence predictions
    high_conf_count = sum(confidences >= args.threshold)
    if high_conf_count > 0:
        print(f"\n{high_conf_count} predictions with confidence >= {args.threshold}")
        if high_conf_count <= 10:  # Show details for small numbers
            high_conf_df = output_df[output_df['confidence'] >= args.threshold]
            print("\nHigh Confidence Predictions:")
            for idx, row in high_conf_df.iterrows():
                print(f"  Patient {idx+1}: {row['risk_level']} (Confidence: {row['confidence']:.2%})")
    
    print(f"\n✅ Batch prediction completed successfully!")
    print(f"   Total processed: {len(input_df)} patients")
    print(f"   High risk: {sum(predictions)} patients")
    print(f"   Low risk: {len(predictions) - sum(predictions)} patients")

if __name__ == "__main__":
    main()