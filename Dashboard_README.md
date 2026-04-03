# 🏥 Diabetes Prediction Dashboard

An interactive Streamlit dashboard for diabetes risk prediction using machine learning.

## 🚀 Live Demo

[View Dashboard on Streamlit Cloud](https://your-streamlit-app-link.streamlit.app) *(Deploy after creating)*

## 📊 Features

### 🏠 Overview
- Dataset statistics and model performance summary
- Clinical features description
- Key metrics and performance indicators

### 📈 Model Performance
- Confusion matrix visualization
- ROC curve with AUC score
- Detailed classification report
- Cross-validation results

### 🔍 Feature Analysis
- Feature importance ranking (Random Forest)
- Interactive feature distribution plots
- Clinical interpretation of feature contributions

### 🎯 Make Prediction
- Interactive patient data input form
- Real-time diabetes risk prediction
- Probability scores and risk assessment
- Feature contribution breakdown

### 📋 Data Explorer
- Dataset summary statistics
- Class distribution visualization
- Sample data viewer
- Statistical analysis

## 🛠️ Technical Details

- **Model**: Random Forest Classifier (100 trees)
- **Accuracy**: 98.25% on test set
- **ROC-AUC**: 1.00 (perfect discrimination)
- **Features**: 8 clinical parameters
- **Dataset**: 2,000 patient records

## 🏃‍♀️ Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

3. **Open your browser** to `http://localhost:8501`

## 📦 Dependencies

- streamlit: Web app framework
- pandas: Data manipulation
- scikit-learn: Machine learning
- matplotlib/seaborn: Visualization
- plotly: Interactive charts
- joblib: Model serialization

## 🎯 Usage for LinkedIn

This dashboard is perfect for showcasing on LinkedIn:

1. **Deploy to Streamlit Cloud** (free tier available)
2. **Share the live link** in your post
3. **Highlight key achievements:**
   - 98.25% model accuracy
   - Clinical-grade diabetes prediction
   - Interactive data science project
   - End-to-end ML pipeline

## 📊 Model Performance Highlights

- **Test Accuracy**: 98.25%
- **Cross-Validation**: 99.45% (5-fold)
- **ROC-AUC**: 1.00
- **Precision**: 97% for diabetes detection
- **Recall**: 98% for diabetes detection

## 🔬 Clinical Features

The model uses 8 evidence-based clinical features:
- Pregnancies (gestational diabetes risk)
- Glucose (primary diagnostic marker)
- Blood Pressure (cardiometabolic risk)
- Skin Thickness (body composition)
- Insulin (metabolic marker)
- BMI (weight-related risk)
- Diabetes Pedigree Function (genetic risk)
- Age (progressive risk factor)

## 👩‍⚕️ Clinical Relevance

This dashboard demonstrates how machine learning can support clinical decision-making in diabetes prevention and management, providing healthcare professionals with data-driven risk assessment tools.

## 📝 License

Built for educational and demonstration purposes.

## 🤝 Connect

Feel free to connect on LinkedIn to discuss this project or collaborate on healthcare AI initiatives!

---

**Built by:** Salma Godiya Issifu, RD
**Date:** April 2026
**GitHub:** [salmaissifu4-hash/diabetes_prediction](https://github.com/salmaissifu4-hash/diabetes_prediction)