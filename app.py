"""
Heart Disease Prediction Streamlit App
This app provides a web interface for heart disease prediction using the trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run model_training.py first!")
        return None, None

def create_directories():
    """Create necessary directories"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

def clean_uploaded_csv(df):
    """
    Clean uploaded CSV to match model expectations
    Automatically handles target columns and missing features
    """
    # Expected feature columns (based on the training data)
    expected_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Remove target/condition columns if they exist
    target_columns = ['target', 'condition', 'heart_disease', 'label', 'class', 'output', 'result']
    removed_targets = []
    
    for col in target_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
            removed_targets.append(col)
    
    if removed_targets:
        st.success(f"✅ Automatically removed target column(s): {', '.join(removed_targets)}")
        st.info("ℹ️ These columns were removed because they represent what we're trying to predict!")
    
    # Check if all expected features are present
    missing_features = [col for col in expected_features if col not in df.columns]
    if missing_features:
        st.error(f"❌ Missing required columns: {', '.join(missing_features)}")
        st.info("Please ensure your CSV contains these columns:")
        
        # Show expected columns in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Required columns:**")
            for i, col in enumerate(expected_features[:7]):
                st.write(f"• {col}")
        with col2:
            st.write("**Continued:**")
            for col in expected_features[7:]:
                st.write(f"• {col}")
        
        return None
    
    # Remove any extra columns that aren't needed
    extra_columns = [col for col in df.columns if col not in expected_features]
    if extra_columns:
        st.info(f"ℹ️ Removed extra columns: {', '.join(extra_columns)}")
        st.write("These columns were not needed for prediction.")
    
    # Keep only the expected features in the correct order
    df_cleaned = df[expected_features].copy()
    
    # Validate data types and ranges
    validation_issues = validate_csv_data(df_cleaned)
    
    return df_cleaned

def validate_csv_data(df):
    """
    Validate the CSV data for common issues
    """
    issues = []
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("⚠️ Found missing values in the following columns:")
        for col, count in missing_values[missing_values > 0].items():
            st.write(f"• {col}: {count} missing values")
        st.info("Missing values will be handled automatically during prediction.")
        issues.append("missing_values")
    
    # Check data types for numeric columns
    numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(f"⚠️ Column '{col}' should be numeric but contains non-numeric values")
                issues.append(f"non_numeric_{col}")
    
    # Check value ranges (basic validation)
    if 'age' in df.columns:
        age_issues = df[(df['age'] < 0) | (df['age'] > 120)]
        if len(age_issues) > 0:
            st.warning(f"⚠️ Found {len(age_issues)} rows with unusual age values (outside 0-120 range)")
            issues.append("age_range")
    
    if 'trestbps' in df.columns:
        bp_issues = df[(df['trestbps'] < 50) | (df['trestbps'] > 300)]
        if len(bp_issues) > 0:
            st.warning(f"⚠️ Found {len(bp_issues)} rows with unusual blood pressure values")
            issues.append("bp_range")
    
    if 'chol' in df.columns:
        chol_issues = df[(df['chol'] < 50) | (df['chol'] > 800)]
        if len(chol_issues) > 0:
            st.warning(f"⚠️ Found {len(chol_issues)} rows with unusual cholesterol values")
            issues.append("chol_range")
    
    # Check categorical columns
    categorical_ranges = {
        'sex': [0, 1],
        'cp': [0, 1, 2, 3],
        'fbs': [0, 1],
        'restecg': [0, 1, 2],
        'exang': [0, 1],
        'slope': [0, 1, 2],
        'ca': [0, 1, 2, 3, 4],
        'thal': [0, 1, 2, 3]
    }
    
    for col, valid_values in categorical_ranges.items():
        if col in df.columns:
            invalid_values = df[~df[col].isin(valid_values)]
            if len(invalid_values) > 0:
                st.warning(f"⚠️ Column '{col}' contains invalid values. Expected: {valid_values}")
                issues.append(f"invalid_{col}")
    
    if not issues:
        st.success("✅ Data validation passed! Your CSV looks good.")
    
    return issues

def save_prediction_log(user_data, prediction, probability):
    """Save prediction to CSV log"""
    log_file = 'logs/prediction_log.csv'
    
    # Prepare log data
    log_data = user_data.copy()
    log_data['prediction'] = prediction
    log_data['probability'] = probability
    log_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert to DataFrame
    log_df = pd.DataFrame([log_data])
    
    # Append to existing file or create new one
    if os.path.exists(log_file):
        existing_df = pd.read_csv(log_file)
        updated_df = pd.concat([existing_df, log_df], ignore_index=True)
        updated_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, index=False)

def generate_pdf_report(user_data, prediction, probability, feature_importance=None):
    """Generate PDF report of the prediction"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Heart Disease Prediction Report", title_style))
    story.append(Spacer(1, 20))
    
    # Prediction Result
    result_text = "HIGH RISK" if prediction == 1 else "LOW RISK"
    result_color = colors.red if prediction == 1 else colors.green
    
    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=result_color,
        alignment=1
    )
    story.append(Paragraph(f"Prediction: {result_text}", result_style))
    story.append(Paragraph(f"Confidence: {probability:.2%}", result_style))
    story.append(Spacer(1, 20))
    
    # Patient Data
    story.append(Paragraph("Patient Information", styles['Heading2']))
    
    # Create table for patient data
    feature_names = {
        'age': 'Age (years)',
        'sex': 'Sex (0: Female, 1: Male)',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'restecg': 'Resting ECG Results',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'Slope of Peak Exercise ST Segment',
        'ca': 'Number of Major Vessels',
        'thal': 'Thalassemia'
    }
    
    table_data = [['Feature', 'Value']]
    for key, value in user_data.items():
        if key in feature_names:
            table_data.append([feature_names[key], str(value)])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    # Recommendations
    story.append(Spacer(1, 20))
    story.append(Paragraph("Recommendations", styles['Heading2']))
    
    if prediction == 1:
        recommendations = [
            "Consult with a cardiologist immediately",
            "Follow a heart-healthy diet low in saturated fats",
            "Engage in regular, moderate exercise as approved by your doctor",
            "Monitor blood pressure and cholesterol levels regularly",
            "Consider stress management techniques",
            "Avoid smoking and limit alcohol consumption"
        ]
    else:
        recommendations = [
            "Continue maintaining a healthy lifestyle",
            "Regular check-ups with your healthcare provider",
            "Maintain a balanced diet rich in fruits and vegetables",
            "Stay physically active with regular exercise",
            "Monitor your health metrics periodically",
            "Avoid risk factors like smoking and excessive alcohol"
        ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    
    # Disclaimer
    story.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(
        "Disclaimer: This prediction is for informational purposes only and should not replace professional medical advice. "
        "Please consult with a qualified healthcare provider for proper diagnosis and treatment.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    """Main Streamlit app"""
    
    # Create directories
    create_directories()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Please run `python model_training.py` first to train the model!")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Single Prediction", "Batch Prediction", "Data Visualization", "Prediction History"])
    
    if page == "Single Prediction":
        single_prediction_page(model, scaler)
    elif page == "Batch Prediction":
        batch_prediction_page(model, scaler)
    elif page == "Data Visualization":
        data_visualization_page()
    elif page == "Prediction History":
        prediction_history_page()

def single_prediction_page(model, scaler):
    """Single prediction page"""
    st.markdown('<h2 class="sub-header">Single Patient Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        
        st.subheader("Chest Pain & Heart Rate")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.subheader("Blood Pressure & Blood Sugar")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col2:
        st.subheader("Cholesterol & ECG")
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], 
                              format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"][x])
        
        st.subheader("Exercise Test Results")
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], 
                           format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        
        st.subheader("Advanced Measurements")
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                          format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
    
    # Prediction button
    if st.button("🔍 Predict Heart Disease Risk", type="primary"):
        # Prepare input data
        user_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        # Create DataFrame and scale
        input_df = pd.DataFrame([user_data])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        confidence = max(probability)
        
        # Display results
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box high-risk">
                <h3>⚠️ HIGH RISK of Heart Disease</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p>Please consult with a healthcare professional immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box low-risk">
                <h3>✅ LOW RISK of Heart Disease</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p>Continue maintaining a healthy lifestyle!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Save to log
        save_prediction_log(user_data, prediction, confidence)
        
        # Generate and offer PDF download
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Generate PDF Report"):
                pdf_buffer = generate_pdf_report(user_data, prediction, confidence)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col2:
            st.success("✅ Prediction saved to log!")

def batch_prediction_page(model, scaler):
    """Batch prediction page with automatic CSV cleaning"""
    st.markdown('<h2 class="sub-header">Batch Prediction</h2>', unsafe_allow_html=True)
    
    st.info("📁 Upload a CSV file with patient data for batch prediction. "
           "The app will automatically handle column formatting!")
    
    # Show expected format
    with st.expander("📋 Click to see expected CSV format"):
        st.write("Your CSV should contain these columns (in any order):")
        expected_cols = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            for col in expected_cols[:7]:
                st.write(f"• **{col}**")
        with col2:
            for col in expected_cols[7:]:
                st.write(f"• **{col}**")
        
        st.write("**Note:** If your CSV contains target columns like 'condition', 'target', etc., they will be automatically removed.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            
            st.write("### 📊 Original Uploaded Data Preview")
            st.dataframe(df.head())
            st.write(f"**Original shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Clean the data automatically
            st.write("### 🧹 Data Cleaning Process")
            df_cleaned = clean_uploaded_csv(df)
            
            if df_cleaned is not None:
                st.write("### ✅ Cleaned Data Preview")
                st.dataframe(df_cleaned.head())
                st.write(f"**Cleaned shape:** {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
                
                if st.button("🔍 Run Batch Prediction", type="primary"):
                    with st.spinner("Making predictions..."):
                        try:
                            # Handle missing values if any
                            if df_cleaned.isnull().sum().sum() > 0:
                                st.info("🔧 Handling missing values with forward fill...")
                                df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
                            
                            # Make predictions
                            input_scaled = scaler.transform(df_cleaned)
                            predictions = model.predict(input_scaled)
                            probabilities = model.predict_proba(input_scaled)
                            confidences = np.max(probabilities, axis=1)
                            
                            # Add results to dataframe
                            results_df = df_cleaned.copy()
                            results_df['prediction'] = predictions
                            results_df['risk_level'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                            results_df['confidence'] = confidences
                            results_df['confidence_percent'] = (confidences * 100).round(1)
                            
                            st.success("🎉 Predictions completed successfully!")
                            
                            # Summary statistics
                            st.write("### 📈 Prediction Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Patients", len(results_df))
                            with col2:
                                high_risk_count = int(sum(predictions))
                                st.metric("High Risk", high_risk_count, 
                                         delta=f"{high_risk_count/len(predictions)*100:.1f}%")
                            with col3:
                                low_risk_count = int(len(predictions) - sum(predictions))
                                st.metric("Low Risk", low_risk_count,
                                         delta=f"{low_risk_count/len(predictions)*100:.1f}%")
                            with col4:
                                avg_confidence = confidences.mean()
                                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                            
                            # Show results
                            st.write("### 📋 Detailed Prediction Results")
                            
                            # Display with risk level coloring
                            def highlight_risk(row):
                                if row['risk_level'] == 'High Risk':
                                    return ['background-color: #ffebee'] * len(row)
                                else:
                                    return ['background-color: #e8f5e8'] * len(row)
                            
                            display_df = results_df[['age', 'sex', 'risk_level', 'confidence_percent']].copy()
                            st.dataframe(
                                display_df.style.apply(highlight_risk, axis=1),
                                use_container_width=True
                            )
                            
                            # Full results download
                            st.write("### 📥 Download Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download full results
                                csv_full = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Full Results CSV",
                                    data=csv_full,
                                    file_name=f"batch_predictions_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # Download summary only
                                summary_df = results_df[['age', 'sex', 'risk_level', 'confidence_percent']].copy()
                                csv_summary = summary_df.to_csv(index=False)
                                st.download_button(
                                    label="📋 Download Summary CSV",
                                    data=csv_summary,
                                    file_name=f"batch_predictions_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            # Risk distribution chart
                            st.write("### 📊 Risk Distribution")
                            risk_counts = results_df['risk_level'].value_counts()
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            risk_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                                           colors=['lightcoral', 'lightgreen'])
                            ax.set_ylabel('')
                            ax.set_title('Risk Level Distribution')
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"❌ Error during batch prediction: {str(e)}")
                            st.info("Please check your data format and try again.")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")

def data_visualization_page():
    """Data visualization page"""
    st.markdown('<h2 class="sub-header">Data Visualization</h2>', unsafe_allow_html=True)
    
    # Load the dataset if available
    try:
        df = pd.read_csv('data/heart_disease.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Total Records:** {len(df)}")
            st.write(f"**Features:** {len(df.columns) - 1}")
            st.write(f"**Target Distribution:**")
            target_counts = df['target'].value_counts()
            st.write(f"- No Disease: {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
            st.write(f"- Disease: {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
        
        with col2:
            st.subheader("Heart Disease Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            target_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax.set_ylabel('')
            ax.set_title('Heart Disease Distribution')
            plt.legend(['No Disease', 'Heart Disease'])
            st.pyplot(fig)
        
        # Age distribution
        st.subheader("Age Distribution by Heart Disease Status")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create separate histograms for each class
        no_disease = df[df['target'] == 0]['age']
        disease = df[df['target'] == 1]['age']
        
        ax.hist(no_disease, alpha=0.7, label='No Disease', bins=20, color='lightgreen')
        ax.hist(disease, alpha=0.7, label='Heart Disease', bins=20, color='lightcoral')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution by Heart Disease Status')
        ax.legend()
        st.pyplot(fig)
        
        # Gender analysis
        st.subheader("Heart Disease by Gender")
        gender_disease = pd.crosstab(df['sex'], df['target'])
        gender_disease.index = ['Female', 'Male']
        gender_disease.columns = ['No Disease', 'Heart Disease']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        gender_disease.plot(kind='bar', ax=ax, color=['lightgreen', 'lightcoral'])
        ax.set_title('Heart Disease Distribution by Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.legend()
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        
        # Chest pain analysis
        st.subheader("Chest Pain Type Analysis")
        cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        cp_disease = pd.crosstab(df['cp'], df['target'])
        cp_disease.index = cp_labels
        cp_disease.columns = ['No Disease', 'Heart Disease']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        cp_disease.plot(kind='bar', ax=ax, color=['lightgreen', 'lightcoral'])
        ax.set_title('Heart Disease Distribution by Chest Pain Type')
        ax.set_xlabel('Chest Pain Type')
        ax.set_ylabel('Count')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error("Dataset not found. Please run model_training.py first to generate the dataset.")

def prediction_history_page():
    """Prediction history page"""
    st.markdown('<h2 class="sub-header">Prediction History</h2>', unsafe_allow_html=True)
    
    log_file = 'logs/prediction_log.csv'
    
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        
        if len(df) > 0:
            st.write(f"**Total Predictions:** {len(df)}")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                high_risk_count = sum(df['prediction'] == 1)
                st.metric("High Risk Predictions", high_risk_count)
            with col2:
                low_risk_count = sum(df['prediction'] == 0)
                st.metric("Low Risk Predictions", low_risk_count)
            with col3:
                avg_age = df['age'].mean()
                st.metric("Average Age", f"{avg_age:.1f}")
            with col4:
                avg_confidence = df['probability'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
            
            # Show recent predictions
            st.subheader("Recent Predictions")
            recent_df = df.tail(10).copy()
            recent_df['risk_level'] = recent_df['prediction'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')
            recent_df['probability_percent'] = (recent_df['probability'] * 100).round(1)
            
            # Display with better formatting
            display_columns = ['timestamp', 'age', 'sex', 'risk_level', 'probability_percent']
            column_config = {
                'timestamp': 'Date & Time',
                'age': 'Age',
                'sex': st.column_config.SelectboxColumn(
                    'Gender',
                    options=[0, 1],
                    format_func=lambda x: 'Female' if x == 0 else 'Male'
                ),
                'risk_level': 'Risk Level',
                'probability_percent': st.column_config.NumberColumn(
                    'Confidence %',
                    format="%.1f%%"
                )
            }
            
            st.dataframe(
                recent_df[display_columns],
                column_config=column_config,
                hide_index=True
            )
            
            # Prediction trends
            if len(df) > 1:
                st.subheader("Prediction Trends")
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily_counts = df.groupby(['date', 'prediction']).size().unstack(fill_value=0)
                
                if len(daily_counts) > 1:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    daily_counts.plot(kind='bar', stacked=True, ax=ax, color=['lightgreen', 'lightcoral'])
                    ax.set_title('Daily Prediction Counts')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Number of Predictions')
                    ax.legend(['Low Risk', 'High Risk'])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Download full history
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full History",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No predictions found in the log.")
    else:
        st.info("No prediction history available. Make some predictions first!")

if __name__ == "__main__":
    main()