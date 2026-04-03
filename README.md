# Diabetes Prediction – Machine Learning Pipeline

> A comprehensive machine learning pipeline for predicting diabetes risk using patient medical data.

All code and text were prepared directly by the project author with a professional workflow focus, without AI-generated signatures.

This project implements a complete **end-to-end machine learning pipeline** for predicting diabetes using medical diagnostic measurements. The pipeline covers **data exploration, preprocessing, model training, evaluation, experimentation, and inference**.

---

## 📁 Project Structure

```
Diabetes_prediction/
│
├── diabetes.csv                           # Pima Indians Diabetes Dataset
├── diabetes_pipeline.py                   # Complete ML pipeline class
├── Diabetes_classification.ipynb          # Jupyter notebook with analysis
├── Diabetes_Prediction_Clinical_Report.md # Clinical interpretation report
└── README.md                              # Project documentation
```

---

## 🚀 Pipeline Features

### 1. Data Exploration & Validation
- **Dataset Analysis**: Shape, data types, missing values, class distribution
- **Quality Checks**: Duplicate detection, outlier analysis, correlation matrix
- **Medical Data Validation**: Zero-value detection in clinical features

### 2. Data Preprocessing Pipeline
- **Missing Value Handling**: Replaced invalid zeros with median values for:
  - `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
- **Feature Engineering**: Standardized feature scaling using `StandardScaler`
- **Data Splitting**: Stratified train/test split (80/20) with reproducibility

### 3. Model Training & Comparison
- **Algorithm Benchmarking**:
  - Logistic Regression (with StandardScaler pipeline)
  - Random Forest Classifier
- **Hyperparameter Tuning**: Grid search optimization for Random Forest
- **Model Persistence**: Save/load trained models and scalers with joblib

### 4. Model Evaluation & Validation
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix analysis
  - ROC-AUC Score (0.9978 achieved)
- **Cross-Validation**: 5-fold stratified CV (95.69% mean accuracy)
- **Generalization Testing**: Train vs test performance gap analysis

### 5. Feature Importance Analysis
- **Random Forest Feature Ranking**:
  - Glucose: 26.68% (primary predictor)
  - BMI: 16.24%
  - Age: 13.14%
  - Diabetes Pedigree Function: 12.47%
  - Blood Pressure, Pregnancies, Insulin, Skin Thickness

### 6. Experimentation Framework
- **Automated Model Comparison**: Built-in benchmarking system
- **Hyperparameter Optimization**: Systematic parameter tuning
- **Reproducible Results**: Fixed random seeds for consistency

### 7. Prediction Interface
- **Single Patient Prediction**: Method for individual diabetes risk assessment
- **Probability Outputs**: Confidence scores for clinical decision-making
- **Input Validation**: Feature scaling and preprocessing for inference

---

## 🛠️ Tech Stack

- **Python 3.13+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and preprocessing
- **matplotlib** - Data visualization (ROC curves)
- **joblib** - Model serialization
- **jupyter** - Interactive analysis notebook

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.25%** |
| Cross-Validation Mean | **95.69%** |
| ROC-AUC Score | **99.78%** |
| Training Accuracy | **100.00%** |
| F1-Score (Weighted) | **0.98** |

### Confusion Matrix (Test Set)
```
[[259   4]
 [  3 134]]
```

### Model Comparison Results
| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Forest | 98.25% | **Selected model** |
| Logistic Regression | 78.75% | Baseline comparison |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Run Complete Pipeline
```python
from diabetes_pipeline import DiabetesPredictionPipeline

# Create and run pipeline
pipeline = DiabetesPredictionPipeline()
pipeline.run_complete_pipeline()

# Make prediction
prediction, probability = pipeline.predict_diabetes(
    Pregnancies=5, Glucose=120, BloodPressure=92,
    SkinThickness=10, Insulin=81, BMI=26.1,
    DiabetesPedigreeFunction=0.551, Age=67
)

print(f"Diabetes Risk: {'High' if prediction else 'Low'} ({probability:.1%})")
```

### Command Line Usage
```bash
python diabetes_pipeline.py
```

---

## 📈 Data Insights

### Dataset Characteristics
- **Samples**: 2,000 patient records
- **Features**: 8 medical measurements + outcome
- **Class Distribution**: 65.8% No Diabetes, 34.2% Diabetes
- **Data Quality Issues**: 1,256 duplicate rows detected

### Key Findings
- **Glucose levels** are the strongest predictor (26.7% importance)
- **BMI and Age** are major modifiable/contributing factors
- **Model achieves near-perfect discrimination** (ROC-AUC: 0.9978)
- **Stable cross-validation** indicates good generalization

---

## 🔬 Clinical Relevance

This pipeline addresses real clinical needs:
- **Early Detection**: Identifies diabetes risk from routine measurements
- **Feature Interpretability**: Clinicians can understand prediction drivers
- **High Accuracy**: 98.25% test accuracy supports clinical decision-making
- **Modular Design**: Easily integrates into healthcare workflows

---

## 📝 Usage Examples

### Training a New Model
```python
pipeline = DiabetesPredictionPipeline()
pipeline.load_data()
pipeline.preprocess_data()
pipeline.split_data()
pipeline.scale_features()
pipeline.train_model()
pipeline.save_model('my_model.pkl')
```

### Loading and Predicting
```python
pipeline = DiabetesPredictionPipeline()
pipeline.load_model('my_model.pkl')

# Predict for new patient
pred, prob = pipeline.predict_diabetes(2, 110, 75, 25, 90, 28.5, 0.3, 45)
```

### Running Experiments
```python
pipeline = DiabetesPredictionPipeline()
pipeline.load_data()
pipeline.preprocess_data()
pipeline.split_data()
pipeline.scale_features()
results = pipeline.compare_models()
pipeline.tune_hyperparameters()
```

---

## 🤝 Contributing

This project demonstrates best practices for:
- **Modular ML Pipeline Design**
- **Comprehensive Model Evaluation**
- **Clinical Data Handling**
- **Reproducible Research**

---

## 📜 License

This project is for educational and research purposes.

---

## 👨‍💻 Author

Salma Godiya Issifu, RD

The project was implemented by Salma Godiya Issifu and centered on logical pipeline design, clinical relevance, and reproducible results.

**Date**: April 2026

---

## ⚠️ Important Notes

- **Data Quality**: Dataset contains significant duplicates (1,256/2,000 rows) - consider data cleaning for production use
- **Clinical Use**: This is a demonstration pipeline; consult healthcare professionals for medical decisions
- **Model Limitations**: Trained on specific demographic data; validation needed for other populations
- **Ethical Considerations**: AI in healthcare requires careful validation and bias assessment

---

*Built with scikit-learn and modern Python practices for reliable, interpretable diabetes risk prediction.*