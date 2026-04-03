import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesPredictionPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def load_data(self, filepath='diabetes.csv'):
        """Load the diabetes dataset"""
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded with shape: {self.df.shape}")
        return self.df

    def preprocess_data(self):
        """Handle missing values and prepare features"""
        # Replace 0 values with median for certain columns (as per common practice for this dataset)
        columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in columns_to_replace:
            self.df[col] = self.df[col].replace(0, self.df[col].median())

        # Define features and target
        self.feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        self.X = self.df[self.feature_columns]
        self.y = self.df['Outcome']

        print("Data preprocessed successfully")
        return self.X, self.y

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print(f"Data split: Train shape {self.X_train.shape}, Test shape {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        """Scale the features using StandardScaler"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("Features scaled successfully")
        return self.X_train_scaled, self.X_test_scaled

    def train_model(self):
        """Train the Random Forest model"""
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_trained = True
        print("Model trained successfully")

        # Calculate training accuracy
        train_pred = self.model.predict(self.X_train_scaled)
        train_accuracy = accuracy_score(self.y_train, train_pred)
        print(f"Training accuracy: {train_accuracy:.4f}")

    def evaluate_model(self):
        """Evaluate the model on test set"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Make predictions
        self.y_pred = self.model.predict(self.X_test_scaled)
        self.y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]

        # Calculate metrics
        test_accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        return test_accuracy, self.y_pred, self.y_pred_proba

    def predict_diabetes(self, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
        """Predict diabetes for a single patient"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Create input array
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Scale the input
        input_scaled = self.scaler.transform(input_data)

        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0][1]

        return prediction, probability

    def explore_data(self):
        """Perform basic exploratory data analysis"""
        print("=== Data Exploration ===")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:\n{self.df.dtypes}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        print(f"Class distribution:\n{self.df['Outcome'].value_counts(normalize=True)}")
        print(f"Descriptive statistics:\n{self.df.describe()}")

        # Check for zero values in medical columns
        medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        print(f"Zero values in medical columns:\n{(self.df[medical_cols] == 0).sum()}")

    def validate_data(self):
        """Validate data quality and check for issues"""
        print("=== Data Validation ===")

        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")

        # Check for outliers using IQR
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"{col}: {outliers} outliers")

        # Check correlations
        corr_matrix = self.df.corr()
        print(f"Correlation with target:\n{corr_matrix['Outcome'].sort_values(ascending=False)}")

    def compare_models(self):
        """Compare Logistic Regression and Random Forest models"""
        print("=== Model Comparison ===")

        # Define models
        models = {
            'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = {}

        for name, model in models.items():
            # Train model
            model.fit(self.X_train_scaled if 'Logistic' in name else self.X_train, self.y_train)

            # Predict
            y_pred = model.predict(self.X_test_scaled if 'Logistic' in name else self.X_test)

            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.4f}")

        # Select best model
        best_model = max(results, key=results.get)
        print(f"Best model: {best_model} with accuracy {results[best_model]:.4f}")

        if best_model == 'Random Forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

        return results

    def cross_validate_model(self, cv=5):
        """Perform cross-validation on the model"""
        print("=== Cross-Validation ===")

        if isinstance(self.model, RandomForestClassifier):
            scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        else:
            # For pipeline
            scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring='accuracy')

        print(f"CV Scores: {scores}")
        print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores

    def tune_hyperparameters(self):
        """Perform hyperparameter tuning for Random Forest"""
        print("=== Hyperparameter Tuning ===")

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def analyze_feature_importance(self):
        """Analyze and display feature importance"""
        print("=== Feature Importance Analysis ===")

        if not isinstance(self.model, RandomForestClassifier):
            print("Feature importance only available for Random Forest models")
            return

        # Get feature importance
        feature_importance = self.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        print("Feature Importance:")
        for idx, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")

        return importance_df

    def plot_roc_curve(self):
        """Plot ROC curve and calculate AUC"""
        print("=== ROC Curve Analysis ===")

        if not hasattr(self, 'y_pred_proba'):
            self.evaluate_model()

        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)

        print(f"ROC-AUC Score: {auc_score:.4f}")

        # Plot ROC curve (only if in interactive environment)
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.show()
        except:
            print("Plotting not available in this environment")

        return auc_score

    def run_complete_pipeline(self):
        """Run the complete pipeline with all steps"""
        print("Starting Complete Diabetes Prediction Pipeline...")

        # 1. Load data
        self.load_data()

        # 2. Explore data
        self.explore_data()

        # 3. Validate data
        self.validate_data()

        # 4. Preprocess
        self.preprocess_data()

        # 5. Split
        self.split_data()

        # 6. Scale (for Logistic Regression comparison)
        self.scale_features()

        # 7. Compare models
        self.compare_models()

        # 8. Cross-validate
        self.cross_validate_model()

        # 9. Tune hyperparameters
        self.tune_hyperparameters()

        # 10. Train final model
        self.train_model()

        # 11. Evaluate
        self.evaluate_model()

        # 12. Feature importance
        self.analyze_feature_importance()

        # 13. ROC analysis
        self.plot_roc_curve()

        print("Complete pipeline finished!")

# Example usage
if __name__ == "__main__":
    # Create pipeline instance
    pipeline = DiabetesPredictionPipeline()

    # Run complete pipeline
    pipeline.run_complete_pipeline()

    # Example prediction
    pred, prob = pipeline.predict_diabetes(5, 120, 92, 10, 81, 26.1, 0.551, 67)
    if pred == 1:
        print('Model prediction: Diabetes detected (class 1).')
    else:
        print('Model prediction: No diabetes detected (class 0).')
    print(f'Predicted probability of diabetes: {prob:.2f}%')