# ❤️ Heart Disease Prediction System

A comprehensive machine learning project for predicting heart disease risk using patient health data. This system includes model training, evaluation, and a user-friendly web interface built with Streamlit.

## 🚀 Features

- **Multiple ML Models**: Trains and compares Logistic Regression, Random Forest, and K-Nearest Neighbors
- **Interactive Web App**: User-friendly Streamlit interface for predictions
- **Batch Processing**: Upload CSV files for multiple patient predictions
- **Data Visualization**: Comprehensive charts and analysis of the dataset
- **PDF Reports**: Generate detailed prediction reports
- **Prediction Logging**: Track all predictions with timestamps
- **Model Persistence**: Save and load trained models automatically

## 📊 Dataset

The project uses the UCI Heart Disease dataset with the following features:

- **age**: Age in years
- **sex**: Gender (0 = Female, 1 = Male)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (0 = No, 1 = Yes)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-4)
- **thal**: Thalassemia (0-3)
- **target**: Heart disease presence (0 = No, 1 = Yes)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd heart-disease-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart_disease.csv          # Dataset (auto-generated if not present)
├── models/
│   ├── model.pkl                  # Trained model
│   ├── scaler.pkl                 # Feature scaler
│   └── model_info.txt             # Model information
├── logs/
│   └── prediction_log.csv         # Prediction history
├── visualizations/
│   ├── eda_plots.png              # Exploratory data analysis plots
│   ├── confusion_matrices/        # Model confusion matrices
│   └── feature_importance.png     # Feature importance plots
├── model_training.py              # Model training script
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎯 Usage

### Step 1: Train the Model

First, run the model training script to create and evaluate models:

```bash
python model_training.py
```

This will:
- Load or generate the heart disease dataset
- Perform exploratory data analysis
- Train multiple machine learning models
- Evaluate and compare model performance
- Save the best-performing model and scaler
- Generate visualization plots

### Step 2: Launch the Web Application

After training, start the Streamlit web app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🖥️ Web Application Features

### 1. Single Prediction
- Input patient data through an intuitive form
- Get real-time heart disease risk predictions
- View confidence scores and risk levels
- Generate and download PDF reports
- All predictions are automatically logged

### 2. Batch Prediction
- Upload CSV files with multiple patient records
- Process hundreds of predictions at once
- Download results as CSV files
- View summary statistics

### 3. Data Visualization
- Heart disease distribution charts
- Age and gender analysis
- Feature correlation heatmaps
- Chest pain type analysis
- Interactive plots and insights

### 4. Prediction History
- View all previous predictions
- Track prediction trends over time
- Download complete prediction history
- Summary statistics and metrics

## 🔬 Model Performance

The system trains and compares three machine learning algorithms:

1. **Logistic Regression**: Linear model with good interpretability
2. **Random Forest**: Ensemble method with feature importance
3. **K-Nearest Neighbors**: Instance-based learning algorithm

The best-performing model (based on F1-score) is automatically selected and saved.

## 📈 Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown
- **Cross-Validation**: 5-fold CV for robust evaluation

## 🎨 Visualizations

The project generates various visualizations:

- **EDA Plots**: Dataset overview and distributions
- **Confusion Matrices**: Model performance visualization
- **Feature Importance**: Most important predictive features
- **Correlation Heatmap**: Feature relationships
- **Prediction Trends**: Historical prediction analysis

## 📄 PDF Reports

Generate comprehensive reports including:
- Patient information summary
- Prediction results with confidence scores
- Personalized health recommendations
- Professional disclaimer and guidance

## 🔧 Customization

### Adding New Features
1. Update the feature list in `model_training.py`
2. Modify the Streamlit form in `app.py`
3. Update the PDF report template if needed

### Changing Models
Add new models in the `train_models()` function in `model_training.py`:

```python
models = {
    'Your New Model': YourModelClass(parameters),
    # ... existing models
}
```

### Styling the Web App
Modify the CSS in `app.py` to customize the appearance:

```python
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
```

## 🚨 Important Notes

- **Medical Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice
- **Data Privacy**: All predictions are stored locally in CSV files
- **Model Accuracy**: Results depend on the quality and representativeness of training data
- **Regular Updates**: Retrain models periodically with new data for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues:

1. **Model files not found**: Run `python model_training.py` first
2. **Missing dependencies**: Install requirements with `pip install -r requirements.txt`
3. **Port already in use**: Use `streamlit run app.py --server.port 8502`
4. **Memory issues**: Reduce dataset size or use simpler models

### Getting Help:

- Check the console output for detailed error messages
- Ensure all dependencies are properly installed
- Verify that all required directories exist
- Make sure you have sufficient disk space for model files

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the documentation and code comments

---

**Built with ❤️ using Python, scikit-learn, and Streamlit**