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
import io

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Safety threshold configuration - TOUGHER THRESHOLD FOR MEDICAL SAFETY
SAFETY_THRESHOLD = 0.3  # Lower = more conservative (more likely to predict high risk)
# 0.5 = default, 0.3 = conservative, 0.2 = very conservative, 0.1 = extremely conservative

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
    .safety-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
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
    log_data['risk_score'] = probability if prediction == 1 else (1 - probability)
    log_data['threshold_used'] = SAFETY_THRESHOLD
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
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        result_style = ParagraphStyle(
            'ResultStyle',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        normal_bold = ParagraphStyle(
            'NormalBold',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        # Title
        story.append(Paragraph("❤️ Heart Disease Prediction Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {current_time}", styles['Normal']))
        story.append(Paragraph(f"<b>Model Used:</b> Machine Learning Classifier (Conservative Mode)", styles['Normal']))
        story.append(Paragraph(f"<b>Safety Threshold:</b> {SAFETY_THRESHOLD:.1%} (Enhanced Safety)", styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Prediction Result
        result_text = "HIGH RISK" if prediction == 1 else "LOW RISK"
        result_color = colors.red if prediction == 1 else colors.green
        
        # Update result style with color
        result_style_colored = ParagraphStyle(
            'ResultStyleColored',
            parent=result_style,
            textColor=result_color
        )
        
        story.append(Paragraph(f"<b>Prediction Result: {result_text}</b>", result_style_colored))
        story.append(Paragraph(f"<b>Confidence Level: {probability:.1%}</b>", result_style_colored))
        story.append(Spacer(1, 30))
        
        # Safety Information
        safety_info = f"""
        <b>Conservative Analysis:</b> This prediction uses a safety threshold of {SAFETY_THRESHOLD:.1%} 
        instead of the standard 50%, making it more likely to identify potential heart disease risk. 
        This approach prioritizes patient safety by reducing the chance of missing a serious condition.
        """
        story.append(Paragraph(safety_info, styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Patient Information Section
        story.append(Paragraph("Patient Information", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Feature descriptions for better readability
        feature_names = {
            'age': 'Age (years)',
            'sex': 'Gender',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure (mm Hg)',
            'chol': 'Serum Cholesterol (mg/dl)',
            'fbs': 'Fasting Blood Sugar > 120 mg/dl',
            'restecg': 'Resting ECG Results',
            'thalach': 'Maximum Heart Rate Achieved',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression Induced by Exercise',
            'slope': 'Slope of Peak Exercise ST Segment',
            'ca': 'Number of Major Vessels (0-4)',
            'thal': 'Thalassemia'
        }
        
        # Value interpretations
        def format_value(key, value):
            if key == 'sex':
                return "Female" if value == 0 else "Male"
            elif key == 'cp':
                cp_types = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
                return f"{value} ({cp_types[int(value)] if 0 <= int(value) <= 3 else 'Unknown'})"
            elif key == 'fbs':
                return "No" if value == 0 else "Yes"
            elif key == 'restecg':
                ecg_types = ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"]
                return f"{value} ({ecg_types[int(value)] if 0 <= int(value) <= 2 else 'Unknown'})"
            elif key == 'exang':
                return "No" if value == 0 else "Yes"
            elif key == 'slope':
                slope_types = ["Upsloping", "Flat", "Downsloping"]
                return f"{value} ({slope_types[int(value)] if 0 <= int(value) <= 2 else 'Unknown'})"
            elif key == 'thal':
                thal_types = ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"]
                return f"{value} ({thal_types[int(value)] if 0 <= int(value) <= 3 else 'Unknown'})"
            else:
                return str(value)
        
        # Create patient information table
        table_data = [['Feature', 'Value']]
        for key, value in user_data.items():
            if key in feature_names:
                formatted_value = format_value(key, value)
                table_data.append([feature_names[key], formatted_value])
        
        # Create and style the table
        table = Table(table_data, colWidths=[3*inch, 2.5*inch])
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # Risk Assessment Section
        story.append(Paragraph("Risk Assessment", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        if prediction == 1:
            risk_explanation = f"""
            Based on the provided health parameters, our conservative machine learning model indicates a 
            <b>HIGH RISK</b> for heart disease. Using a safety threshold of {SAFETY_THRESHOLD:.1%}, this means 
            that the combination of factors in your health profile shows patterns that warrant immediate 
            medical attention and further evaluation.
            """
        else:
            risk_explanation = f"""
            Based on the provided health parameters, our conservative machine learning model indicates a 
            <b>LOW RISK</b> for heart disease. Even with our enhanced safety threshold of {SAFETY_THRESHOLD:.1%}, 
            your health profile shows patterns similar to patients without heart disease. However, 
            continue regular health maintenance and monitoring.
            """
        
        story.append(Paragraph(risk_explanation, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("Health Recommendations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        if prediction == 1:
            recommendations = [
                "🏥 <b>URGENT:</b> Schedule an appointment with a cardiologist or healthcare provider immediately",
                "💊 <b>Medical Care:</b> Follow all prescribed medications and treatment plans strictly",
                "🍎 <b>Diet:</b> Adopt a heart-healthy diet low in saturated fats, trans fats, and sodium",
                "🏃 <b>Exercise:</b> Engage in regular, moderate exercise as approved by your doctor",
                "📊 <b>Monitoring:</b> Monitor blood pressure, cholesterol, and blood sugar levels regularly",
                "🚭 <b>Lifestyle:</b> Avoid smoking and limit alcohol consumption immediately",
                "😌 <b>Stress:</b> Practice stress management techniques like meditation or yoga",
                "💤 <b>Sleep:</b> Maintain regular sleep patterns (7-9 hours per night)",
                "🔄 <b>Follow-up:</b> Schedule regular check-ups and follow-up tests as recommended"
            ]
        else:
            recommendations = [
                "✅ <b>Maintain:</b> Continue your current healthy lifestyle practices",
                "🏥 <b>Check-ups:</b> Schedule regular preventive check-ups with your healthcare provider",
                "🥗 <b>Diet:</b> Maintain a balanced diet rich in fruits, vegetables, and whole grains",
                "🏃 <b>Exercise:</b> Stay physically active with at least 150 minutes of moderate exercise weekly",
                "📊 <b>Monitoring:</b> Monitor your health metrics periodically",
                "🚭 <b>Prevention:</b> Continue avoiding risk factors like smoking and excessive alcohol",
                "💤 <b>Sleep:</b> Maintain good sleep hygiene and regular sleep schedule",
                "😌 <b>Stress:</b> Continue managing stress through healthy activities",
                "⚠️ <b>Vigilance:</b> Remain alert to any changes in your health status"
            ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 30))
        
        # Important Notes Section
        story.append(Paragraph("Important Notes", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        important_notes = [
            f"<b>Conservative Analysis:</b> This uses a {SAFETY_THRESHOLD:.1%} threshold for enhanced patient safety.",
            "<b>Model Accuracy:</b> Based on machine learning analysis of health data patterns.",
            "<b>Not a Diagnosis:</b> This tool does not provide medical diagnoses or replace professional medical advice.",
            "<b>Individual Variation:</b> Health outcomes can vary significantly between individuals.",
            "<b>Regular Updates:</b> Health status can change; regular medical check-ups are essential."
        ]
        
        for note in important_notes:
            story.append(Paragraph(f"• {note}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 40))
        
        # Medical Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        disclaimer_text = """
        <b>MEDICAL DISCLAIMER:</b> This prediction report is for informational and educational purposes only. 
        It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition. Never disregard professional medical advice or 
        delay in seeking it because of something you have read in this report.
        """
        
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("Generated by Heart Disease Prediction System - Conservative Mode", footer_style))
        
        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

def main():
    """Main Streamlit app"""
    
    # Create directories
    create_directories()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction</h1>', unsafe_allow_html=True)
    
    # Safety threshold information
    st.markdown(f"""
    <div class="safety-info">
        <h4>🛡️ Enhanced Safety Mode Active</h4>
        <p><strong>Threshold:</strong> {SAFETY_THRESHOLD:.1%} (More conservative than standard 50%)</p>
        <p><strong>Impact:</strong> Higher sensitivity - more likely to identify potential heart disease risk</p>
        <p><strong>Purpose:</strong> Prioritizes patient safety by reducing false negatives</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Safety information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Safety Settings")
    st.sidebar.info(f"**Threshold:** {SAFETY_THRESHOLD:.1%}")
    st.sidebar.write("🔒 **Conservative Mode:** Lower threshold means more patients will be flagged as high-risk for safety.")
    st.sidebar.write("📊 **Impact:** More false positives, fewer false negatives")
    
    if page == "Single Prediction":
        single_prediction_page(model, scaler)
    elif page == "Batch Prediction":
        batch_prediction_page(model, scaler)
    elif page == "Data Visualization":
        data_visualization_page()
    elif page == "Prediction History":
        prediction_history_page()

def single_prediction_page(model, scaler):
    """Single prediction page with session state to prevent refresh issues"""
    st.markdown('<h2 class="sub-header">Single Patient Prediction</h2>', unsafe_allow_html=True)
    
    # Initialize session state for prediction results
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'last_prediction_data' not in st.session_state:
        st.session_state.last_prediction_data = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 20, 100, 50, key="age_slider")
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="sex_select")
        
        st.subheader("Chest Pain & Heart Rate")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                         key="cp_select")
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150, key="thalach_slider")
        exang = st.selectbox("Exercise Induced Angina", [0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes", key="exang_select")
        
        st.subheader("Blood Pressure & Blood Sugar")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, key="trestbps_slider")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes", key="fbs_select")
    
    with col2:
        st.subheader("Cholesterol & ECG")
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240, key="chol_slider")
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], 
                              format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"][x],
                              key="restecg_select")
        
        st.subheader("Exercise Test Results")
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1, key="oldpeak_slider")
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], 
                           format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                           key="slope_select")
        
        st.subheader("Advanced Measurements")
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4], key="ca_select")
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                          format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x],
                          key="thal_select")
    
    # Current input data
    current_user_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    # Check if input has changed (reset prediction state if so)
    if (st.session_state.last_prediction_data is not None and 
        current_user_data != st.session_state.last_prediction_data.get('user_data', {})):
        st.session_state.prediction_made = False
        st.session_state.last_prediction_data = None
    
    # Prediction button
    if st.button("🔍 Predict Heart Disease Risk", type="primary", key="predict_button"):
        # Create DataFrame and scale
        input_df = pd.DataFrame([current_user_data])
        input_scaled = scaler.transform(input_df)
        
        # MODIFIED: Make prediction with custom threshold
        probability = model.predict_proba(input_scaled)[0]
        confidence = max(probability)
        
        # Apply custom safety threshold (tougher)
        prediction = 1 if probability[1] > SAFETY_THRESHOLD else 0
        
        # Store prediction results in session state
        st.session_state.prediction_made = True
        st.session_state.last_prediction_data = {
            'user_data': current_user_data.copy(),
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence
        }
        
        # Save to log
        save_prediction_log(current_user_data, prediction, confidence)
    
    # Display results if prediction has been made
    if st.session_state.prediction_made and st.session_state.last_prediction_data:
        prediction_data = st.session_state.last_prediction_data
        prediction = prediction_data['prediction']
        probability = prediction_data['probability']
        confidence = prediction_data['confidence']
        user_data = prediction_data['user_data']
        
        # Display results
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box high-risk">
                <h3>⚠️ HIGH RISK of Heart Disease</h3>
                <p><strong>Risk Score:</strong> {probability[1]:.2%}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Safety Threshold:</strong> {SAFETY_THRESHOLD:.1%} (Conservative)</p>
                <p><strong>⚕️ Please consult with a healthcare professional immediately.</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box low-risk">
                <h3>✅ LOW RISK of Heart Disease</h3>
                <p><strong>Risk Score:</strong> {probability[1]:.2%}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Safety Threshold:</strong> {SAFETY_THRESHOLD:.1%} (Conservative)</p>
                <p><strong>Continue maintaining a healthy lifestyle!</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional risk interpretation
        if probability[1] > 0.7:
            risk_level = "Very High"
            risk_color = "🔴"
        elif probability[1] > 0.5:
            risk_level = "High"
            risk_color = "🟠"
        elif probability[1] > 0.3:
            risk_level = "Moderate"
            risk_color = "🟡"
        else:
            risk_level = "Low"
            risk_color = "🟢"
        
        st.info(f"{risk_color} **Risk Level:** {risk_level} | **Raw Score:** {probability[1]:.3f}")
        
        # Action buttons section
        st.markdown("---")
        st.subheader("📋 Generate Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate PDF Report button (now won't refresh)
            if st.button("📄 Generate PDF Report", key="generate_pdf_button"):
                with st.spinner("Generating PDF report..."):
                    pdf_buffer = generate_pdf_report(user_data, prediction, confidence)
                    
                    if pdf_buffer is not None:
                        # Store PDF in session state for download
                        st.session_state.pdf_data = pdf_buffer.getvalue()
                        st.session_state.pdf_ready = True
                        st.success("✅ PDF report generated successfully!")
                    else:
                        st.error("❌ Failed to generate PDF report. Please try again.")
        
        with col2:
            # Download PDF button (appears after generation)
            if st.session_state.get('pdf_ready', False):
                st.download_button(
                    label="📥 Download PDF Report",
                    data=st.session_state.pdf_data,
                    file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="pdf_download_button"
                )
        
        with col3:
            # Reset prediction button
            if st.button("🔄 New Prediction", key="reset_button"):
                st.session_state.prediction_made = False
                st.session_state.last_prediction_data = None
                st.session_state.pdf_ready = False
                if 'pdf_data' in st.session_state:
                    del st.session_state.pdf_data
                st.rerun()
        
        st.success("✅ Prediction saved to log!")
        
        # Show input summary
        with st.expander("📊 Input Summary", expanded=False):
            st.json(user_data)

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
                            
                            # MODIFIED: Make predictions with custom threshold
                            input_scaled = scaler.transform(df_cleaned)
                            probabilities = model.predict_proba(input_scaled)
                            
                            # Apply custom safety threshold (tougher)
                            predictions = (probabilities[:, 1] > SAFETY_THRESHOLD).astype(int)
                            confidences = np.max(probabilities, axis=1)
                            
                            # Add results to dataframe
                            results_df = df_cleaned.copy()
                            results_df['prediction'] = predictions
                            results_df['risk_level'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                            results_df['confidence'] = confidences
                            results_df['confidence_percent'] = (confidences * 100).round(1)
                            results_df['risk_score'] = (probabilities[:, 1] * 100).round(1)  # Add risk score
                            results_df['threshold_used'] = SAFETY_THRESHOLD
                            
                            # Display threshold info
                            st.info(f"🛡️ **Safety Threshold Used:** {SAFETY_THRESHOLD:.1%} (Conservative Mode)")
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
                            
                            display_df = results_df[['age', 'sex', 'risk_level', 'risk_score', 'confidence_percent']].copy()
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
                                summary_df = results_df[['age', 'sex', 'risk_level', 'risk_score', 'confidence_percent']].copy()
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
                            ax.set_title(f'Risk Level Distribution (Threshold: {SAFETY_THRESHOLD:.1%})')
                            st.pyplot(fig)
                            
                            # Risk score distribution
                            st.write("### 📈 Risk Score Distribution")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(probabilities[:, 1], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.axvline(x=SAFETY_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                                      label=f'Safety Threshold ({SAFETY_THRESHOLD:.1%})')
                            ax.set_xlabel('Risk Score')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Risk Scores')
                            ax.legend()
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
    """Prediction history page with fixed clear functionality"""
    st.markdown('<h2 class="sub-header">📊 Prediction History</h2>', unsafe_allow_html=True)
    
    # Load prediction history
    log_file = 'logs/prediction_log.csv'
    
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        
        if not df.empty:
            # Transform categorical values for better display
            df_display = df.copy()
            
            # Transform sex column
            if 'sex' in df_display.columns:
                df_display['sex'] = df_display['sex'].map({0: 'Female', 1: 'Male'})
            
            # Transform target/prediction column (if it exists)
            if 'target' in df_display.columns:
                df_display['target'] = df_display['target'].map({0: 'No Heart Disease', 1: 'Heart Disease'})
            if 'prediction' in df_display.columns:
                df_display['prediction'] = df_display['prediction'].map({0: 'Low Risk', 1: 'High Risk'})
            
            # Transform other categorical columns if they exist
            if 'cp' in df_display.columns:
                chest_pain_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
                df_display['cp'] = df_display['cp'].map(chest_pain_map)
            
            if 'fbs' in df_display.columns:
                df_display['fbs'] = df_display['fbs'].map({0: 'No', 1: 'Yes'})
            
            if 'exang' in df_display.columns:
                df_display['exang'] = df_display['exang'].map({0: 'No', 1: 'Yes'})
            
            if 'restecg' in df_display.columns:
                restecg_map = {0: 'Normal', 1: 'ST-T Abnormality', 2: 'LV Hypertrophy'}
                df_display['restecg'] = df_display['restecg'].map(restecg_map)
            
            if 'slope' in df_display.columns:
                slope_map = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
                df_display['slope'] = df_display['slope'].map(slope_map)
            
            if 'thal' in df_display.columns:
                thal_map = {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect', 3: 'Unknown'}
                df_display['thal'] = df_display['thal'].map(thal_map)
            
            # Show summary first
            st.write("### 📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(df_display))
            
            with col2:
                if 'prediction' in df_display.columns:
                    heart_disease_count = len(df_display[df_display['prediction'] == 'High Risk'])
                    st.metric("High Risk Predictions", heart_disease_count)
                elif 'target' in df_display.columns:
                    heart_disease_count = len(df_display[df_display['target'] == 'Heart Disease'])
                    st.metric("Heart Disease Cases", heart_disease_count)
            
            with col3:
                if 'prediction' in df_display.columns:
                    no_disease_count = len(df_display[df_display['prediction'] == 'Low Risk'])
                    st.metric("Low Risk Predictions", no_disease_count)
                elif 'target' in df_display.columns:
                    no_disease_count = len(df_display[df_display['target'] == 'No Heart Disease'])
                    st.metric("No Heart Disease Cases", no_disease_count)
            
            with col4:
                if 'threshold_used' in df_display.columns:
                    threshold_used = df_display['threshold_used'].iloc[0] if not df_display['threshold_used'].empty else SAFETY_THRESHOLD
                    st.metric("Safety Threshold", f"{threshold_used:.1%}")
                else:
                    st.metric("Safety Threshold", f"{SAFETY_THRESHOLD:.1%}")
            
            # Display the dataframe with proper column configuration
            st.write("### 📋 Prediction Records")
            
            # Select most relevant columns for display
            display_columns = ['timestamp', 'age', 'sex', 'prediction', 'risk_score', 'probability', 'threshold_used']
            available_columns = [col for col in display_columns if col in df_display.columns]
            
            if available_columns:
                st.dataframe(
                    df_display[available_columns],
                    column_config={
                        'timestamp': st.column_config.DatetimeColumn('Timestamp'),
                        'age': st.column_config.NumberColumn('Age'),
                        'sex': st.column_config.TextColumn('Gender'),
                        'prediction': st.column_config.TextColumn('Prediction'),
                        'risk_score': st.column_config.NumberColumn('Risk Score'),
                        'probability': st.column_config.NumberColumn('Confidence'),
                        'threshold_used': st.column_config.NumberColumn('Threshold'),
                    },
                    use_container_width=True
                )
            else:
                st.dataframe(df_display, use_container_width=True)
            
            # Action buttons
            st.write("### 🔧 Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download button
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download History",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Clear history button with confirmation
                if st.button("🗑️ Clear History", type="secondary"):
                    st.session_state.show_clear_confirm = True
            
            with col3:
                # Show full details toggle
                if st.button("📊 Show Full Details"):
                    with st.expander("Full Prediction History", expanded=True):
                        st.dataframe(df_display, use_container_width=True)
            
            # Clear confirmation dialog
            if st.session_state.get('show_clear_confirm', False):
                st.warning("⚠️ Are you sure you want to clear all prediction history? This action cannot be undone.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Clear History", type="primary"):
                        try:
                            if os.path.exists(log_file):
                                os.remove(log_file)
                                st.success("🗑️ Prediction history cleared successfully!")
                                st.session_state.show_clear_confirm = False
                                st.rerun()
                            else:
                                st.error("History file not found.")
                        except Exception as e:
                            st.error(f"Error clearing history: {str(e)}")
                
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.show_clear_confirm = False
                        st.rerun()
            
            # Prediction trends visualization
            if 'timestamp' in df_display.columns and len(df_display) > 1:
                st.write("### 📈 Prediction Trends")
                
                # Convert timestamp to datetime
                df_display['timestamp'] = pd.to_datetime(df_display['timestamp'])
                
                # Daily prediction counts
                daily_predictions = df_display.groupby([df_display['timestamp'].dt.date, 'prediction']).size().unstack(fill_value=0)
                
                if not daily_predictions.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    daily_predictions.plot(kind='bar', ax=ax, color=['lightgreen', 'lightcoral'])
                    ax.set_title('Daily Prediction Trends')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Number of Predictions')
                    ax.legend(title='Risk Level')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        else:
            st.info("📝 No prediction history available yet.")
            st.write("Make some predictions first to see the history here!")
    
    else:
        st.info("📁 No prediction history file found.")
        st.write("Make some predictions first to create the history log!")
    
    # Instructions
    with st.expander("ℹ️ About Prediction History"):
        st.write("""
        **What's stored:**
        - All individual predictions with timestamps
        - Patient parameters used for each prediction
        - Risk scores and confidence levels
        - Safety threshold used for each prediction
        
        **Features:**
        - Download complete history as CSV
        - View prediction trends over time
        - Clear history with confirmation
        - Detailed prediction analytics
        
        **Privacy Note:** 
        All data is stored locally on your system and not shared externally.
        """)

if __name__ == "__main__":
    main()