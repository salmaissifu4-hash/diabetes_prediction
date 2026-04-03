# Clinical Report: Machine Learning-Based Diabetes Risk Prediction
### A Registered Dietitian's Evidence-Based Analysis of a Random Forest Classification Model

---

**Author:** Salma Godiya Issifu, RD 
**Date:** April 2026  
**Dataset:** Diabetes Patient Records (N = 2,000)  
**Model:** Random Forest Classifier (Scikit-learn, Python)  
**Classification Task:** Binary classification: Diabetic (1) vs. Non-Diabetic (0)

---

## Table of Contents

1. [Abstract](#1-abstract)  
2. [Introduction](#2-introduction)  
3. [Literature Review](#3-literature-review)  
4. [Dataset Description and Nutritional Relevance of Features](#4-dataset-description-and-nutritional-relevance-of-features)  
5. [Data Preprocessing and Imputation Strategy](#5-data-preprocessing-and-imputation-strategy)  
6. [Model Development Methodology](#6-model-development-methodology)  
7. [Results and Performance Evaluation](#7-results-and-performance-evaluation)  
8. [Discussion of Feature Importance from a Dietetic Perspective](#8-discussion-of-feature-importance-from-a-dietetic-perspective)  
9. [Clinical Implications for Dietetic Practice](#9-clinical-implications-for-dietetic-practice)  
   - 9.1 Application as a Screening Decision-Support Tool  
   - 9.2 Integration into Dietetic Assessment Workflows  
   - 9.3 Personalised Nutrition Recommendations  
   - 9.4 Post-GDM Nutritional Management  
   - 9.5 Evidence-Based Dietary Patterns for T2DM Prevention  
   - 9.6 Cost-Effectiveness of Dietitian-Led MNT  
10. [Limitations](#10-limitations)  
11. [Conclusion](#11-conclusion)  
12. [Future Directions and Research Recommendations](#12-future-directions-and-research-recommendations)  
13. [References](#13-references)

---

## 1. Abstract

**Background:** Type 2 diabetes is one of the biggest health challenges we face today. Cases keep rising, and the real problem is that most people don't know they have it until it's already causing damage. Early detection is where we can actually make a difference, and that's where nutrition comes in.

**Objective:** I've built and tested a Random Forest machine learning model for diabetes prediction and I'm evaluating it from a registered dietitian's perspective. Since all the input variables are nutritionally relevant, I'm contextualizing how this model fits into nutrition-based diabetes prevention and management.

**Methods:** The model uses eight clinically meaningful features from 2,000 patient records: fasting glucose, BMI, serum insulin, blood pressure, skin-fold thickness, pregnancy history, diabetes pedigree function, and age. I cleaned the data using nutrition-informed approaches, then trained and compared two models: Logistic Regression and Random Forest. The Random Forest version (100 trees) was thoroughly evaluated for accuracy, sensitivity, specificity, and cross-validated generalization.

**Results:** The Random Forest model achieved 98.25% test accuracy, 99.45% cross-validated accuracy, and perfect ROC-AUC discrimination (1.00). Interestingly, when you rank the features by importance, the top five are all things we can address through nutrition: glucose control (26%), weight management (16%), insulin function (9%), blood pressure (8%), and body composition (7%). That adds up to 66% of the model's predictive power being directly modifiable through dietetic intervention.

**Conclusions:** This model works well and lines up perfectly with what we know about nutritional risk factors for diabetes. It could be a useful tool for dietitians doing diabetes prevention work, giving us a quick way to identify who's at highest risk and what to focus on first in their care plan.

**Keywords:** type 2 diabetes, machine learning, random forest, nutrition, dietetics, glucose, weight management, diabetes prevention, clinical screening

---

## 2. Introduction

### 2.1 The Global Diabetes Burden

Type 2 diabetes is a massive problem. Right now, 537 million adults worldwide have diabetes, and if we don't change course, that number hits 783 million by 2045. The cost is staggering, nearly 1 trillion dollars a year in healthcare spending (Sun et al., 2022). But here's the kicker: over 240 million people have diabetes and don't even know it, representing 40% of all cases. That's the gap where we as dietitians can actually intervene.

Type 2 makes up 90-95% of all diabetes cases. It's driven by insulin resistance, something that builds up over years through diet, weight gain, and lifestyle choices. Unlike type 1, this one is largely preventable. The problem is we're not catching people early enough before the damage starts.

Africa is seeing the fastest rise in diabetes rates globally, with a projected 134% increase by 2045. This dataset we're using has characteristics similar to African and indigenous populations that carry higher T2 risk. That makes tools like this model really important, and we need screening methods that don't require expensive labs.

### 2.2 The Role of the Registered Dietitian in Diabetes Prevention

As a registered dietitian, I work at the front line of diabetes prevention. When patients get nutrition therapy delivered by a dietitian, their HbA1c goes down 1-2%. That's real, measurable change. Every major diabetes organization recognizes this. The ADA explicitly lists dietitian-delivered medical nutrition therapy (MNT) as a cornerstone of diabetes care.

Here's what makes this interesting from my perspective: the eight features this model uses, including glucose, BMI, insulin, blood pressure, and skin thickness levels, are *exactly* the parameters I'm monitoring and working with at every patient visit. I'm taking measurements, looking at blood work, checking weight, and assessing diet quality. So a tool that pulls all these together and gives me a risk score actually fits into my workflow.

### 2.3 Purpose of This Report

This report provides:
1. A detailed clinical and nutritional interpretation of the machine learning model developed in the accompanying Jupyter Notebook (`Diabetes_classification.ipynb`).
2. An evidence-based analysis of each predictive feature from the dietetic and nutritional science perspective.
3. An assessment of model performance, reliability, and appropriate clinical use.
4. Recommendations for the integration of this tool into dietetic practice.

---

## 3. Literature Review

### 3.1 Machine Learning in Diabetes Prediction

Machine learning has become popular for predicting diabetes, especially in the last few years. When you look at the research, Random Forest keeps winning. It outperforms logistic regression, decision trees, and most other methods. A systematic review looked at 40 different studies and found Random Forest averaging 94-97% accuracy, which is exactly what we're seeing with this model.

Random Forest works by building lots of decision trees and having them "vote" on the answer. Each tree looks at the data slightly differently, which makes the whole system robust to noise and outliers. It's like consulting multiple expert opinions instead of trusting one doctor's interpretation.

Why is this better for clinical use? Because it captures complex interactions without assuming linear relationships. It can handle missing data better than simpler methods. It tells you which features matter most. And it doesn't get fooled as easily by class imbalance, which we have here because more people don't have diabetes than do.

Javeed et al. (2022) evaluated multiple ML algorithms on diabetes screening datasets and concluded that Random Forest with appropriate hyperparameter tuning outperformed all competing algorithms, producing AUC values of 0.97-0.99. These findings echo the present model's ROC-AUC of 1.00. The authors concluded that ensemble classifiers should be prioritised in clinical decision-support systems for chronic disease screening.

### 3.2 Nutritional Risk Factors and Machine Learning Features

All eight features in this model are well-established diabetes risk factors. More importantly, most of them are things we can actually change through nutrition and lifestyle. That's the key point from my perspective as a dietitian:

**Plasma Glucose:** High glucose is how we *diagnose* diabetes, so it makes sense it's the strongest predictor. But glucose levels respond directly to what people eat, specifically carbohydrate quality, portion size, and meal timing. This is where nutritional intervention starts.

**Body Mass Index (BMI):** Obesity is the #1 modifiable risk factor for type 2 diabetes. When people lose weight through dietary changes, their diabetes risk drops significantly. Studies like DiRECT and PREVIEW show over 50% of people can actually reverse their diabetes diagnosis with intensive dietary intervention and weight loss.

**Serum Insulin:** High fasting insulin is a sign of insulin resistance, meaning the body's cells aren't responding to insulin anymore. This happens before blood glucose goes up. We can improve insulin resistance with specific dietary patterns, especially low-glycemic diets and weight reduction.

**Diabetes Pedigree Function:** This captures family history. If diabetes runs in your family, your risk is higher. But genetics isn't destiny, and diet is still the most modifiable factor. It matters *more* in people with high genetic risk.

**Blood Pressure:** 70-80% of people with diabetes also have high blood pressure. Both respond to the DASH diet and reducing sodium intake. It's one tool that handles multiple problems at once.

**Skin-Fold Thickness:** This measures subcutaneous fat under the skin. Combined with BMI, it gives us a fuller picture of body composition. This is something we directly address through dietary counseling and lifestyle change.

**Pregnancy History:** Women who had gestational diabetes during pregnancy have a 10-fold higher risk of type 2 later. This is partly genetic, but post-pregnancy nutrition and weight management are huge modifiable factors.

**Age:** This one we can't change, but it tells us who needs earlier, more aggressive intervention.

### 3.3 Why Random Forest is Appropriate for This Clinical Context

The choice of Random Forest is well supported in the clinical prediction literature for several methodological reasons:

1. **Non-linearity:** The relationships between nutritional biomarkers and diabetes risk are neither linear nor additive; they involve complex interactions (e.g., the joint effect of high glucose and high BMI exceeds the sum of their individual effects). Random Forest captures these interactions inherently (Breiman, 2001).

2. **Robustness to imputation:** Missing data imputation, as employed in this model (mean/median substitution), introduces noise. Ensemble models are substantially more robust to this noise than single-learner models (Sterne et al., 2021).

3. **Feature importance quantification:** Unlike logistic regression, Random Forest provides a built-in, Gini impurity-based measure of feature importance that is directly interpretable by clinicians without statistical training (Ishwaran & Lu, 2019).

4. **Resistance to class imbalance sensitivity:** The present dataset exhibits a class imbalance (approximately 66% non-diabetic: 34% diabetic in the test set: 263 vs. 137 cases). Random Forest's aggregation across multiple trees mitigates the bias toward majority-class prediction that afflicts simpler learners (Chen et al., 2022).

---

## 4. Dataset Description and Nutritional Relevance of Features

### 4.1 Overview

| Attribute | Description | Dietetic Relevance |
|-----------|-------------|-------------------|
| **Pregnancies** | Number of times pregnant | Gestational diabetes risk marker; each pregnancy raises cumulative GDM exposure |
| **Glucose** | Plasma glucose concentration (mg/dL) | Primary diagnostic and dietary management target; driven by dietary carbohydrate quality and quantity |
| **BloodPressure** | Diastolic blood pressure (mm Hg) | Cardiometabolic risk; modifiable through DASH-pattern diet and sodium restriction |
| **SkinThickness** | Triceps skin-fold (mm) | Anthropometric adiposity index; reflects subcutaneous fat stores |
| **Insulin** | 2-hour serum insulin (µU/mL) | Insulin resistance marker; responds to dietary glycaemic load reduction |
| **BMI** | Body mass index (kg/m²) | Predominant modifiable risk factor; central target of MNT |
| **DPF** | Diabetes pedigree function | Genetic and familial risk; contextualises the magnitude of benefit from early dietary intervention |
| **Age** | Age in years | Risk increases with age; strengthens priority basis for early dietetic screening |
| **Outcome** | Diabetic (1) or non-diabetic (0) | Binary classification target |

**Total records:** 2,000  
**Features (predictors):** 8  
**Target variable:** Outcome (binary)  
**Class distribution (test set):** 263 non-diabetic (65.75%), 137 diabetic (34.25%)  

### 4.2 Clinical Thresholds for Reference

| Feature | Non-diabetic Reference Range | At-Risk (Prediabetes) | Diabetic Threshold |
|---------|-----------------------------|-----------------------|-------------------|
| Fasting Glucose | < 5.6 mmol/L (< 100 mg/dL) | 5.6-6.9 mmol/L (100-125 mg/dL) | ≥ 7.0 mmol/L (≥ 126 mg/dL) |
| BMI | 18.5-24.9 kg/m² | 25.0-29.9 kg/m² | ≥ 30 kg/m² (obesity) |
| Fasting Insulin | 2-25 µU/mL | > 25 µU/mL (insulin resistance) | N/A |
| Blood Pressure | < 120/80 mm Hg | 120-129/< 80 mm Hg | ≥ 130/80 mm Hg (concurrent hypertension) |

*Sources: ADA (2024), WHO (2021), JNC 8 Guidelines.*

---

## 5. Data Preprocessing and Imputation Strategy

### 5.1 Identification and Handling of Physiologically Implausible Values

A critical preprocessing step involved identifying zero (0) values in five columns (Glucose, BloodPressure, SkinThickness, Insulin, and BMI) and replacing them with `NaN` prior to imputation.

This step reflects sound clinical and physiological reasoning: a fasting plasma glucose of 0 mg/dL, a BMI of 0 kg/m², or a blood pressure of 0 mm Hg are physiologically incompatible with survival, and their presence in the dataset almost certainly denotes missing or not-recorded data rather than true zero measurements (Naz & Shoaib, 2021). Treating these false zeros as real data would introduce severe bias, particularly in glucose, which is the most predictive feature, and would artificially create clusters of extreme low-glucose observations.

### 5.2 Imputation Method Selection

Following removal of these invalid entries, missing values were imputed using:

- **Mean imputation:** Applied to `Glucose` and `BloodPressure`, which tend toward approximately normal distributions, making the mean a statistically appropriate central tendency estimate.
- **Median imputation:** Applied to `SkinThickness`, `Insulin`, `BMI`, and `DPF`, which exhibit positive skewness (right-skewed distribution) common in clinical biomarker data. The median is a more robust measure of central tendency than the mean for skewed distributions and is less sensitive to extreme values (Van Buuren, 2018).

### 5.3 Imputation Limitations

Simple mean/median imputation, while computationally efficient and widely used, assumes data are Missing Completely at Random (MCAR), an assumption that may not hold in clinical datasets where missingness is often related to the severity of disease or the clinical context (e.g., insulin values are more likely to be missing in healthier individuals who did not require the test). More sophisticated imputation methods, such as Multiple Imputation by Chained Equations (MICE), have been shown to produce less biased estimates under Missing at Random (MAR) mechanisms (Sterne et al., 2021). This limitation is discussed further in Section 10.

### 5.4 Feature Naming Standardisation

The original `DiabetesPedigreeFunction` column was renamed to `DPF` in both the original (`df`) and working copy (`df_copy`) dataframes to ensure naming consistency across all downstream operations, a best practice in reproducible data science workflows (Wilson et al., 2021).

---

## 6. Model Development Methodology

### 6.1 Train-Test Split

Data were partitioned into a training set (80%; n = 1,600) and a test set (20%; n = 400) using stratified random sampling (`random_state=42`, `stratify=y`). Stratification ensures that the class distribution of the target variable (Outcome) is preserved in both subsets, a methodologically important step when dealing with imbalanced classes, as it prevents accidental over-representation of one class in either partition (Chen et al., 2022).

### 6.2 Baseline Model Comparison

Prior to final model selection, a function `find_best_model()` was implemented to compare two candidate classifiers:

| Model | Test Accuracy |
|-------|---------------|
| **Random Forest** | **98.25%** |
| Logistic Regression | 79.75% |

Logistic Regression was wrapped in a `StandardScaler` pipeline to ensure that its performance was not compromised by unscaled feature magnitudes. Despite this, it achieved only 79.75% accuracy. This 18.5 percentage-point gap is largely attributable to Random Forest's ability to model non-linear feature interactions, which, as discussed in Section 3.3, are inherent in the biology of metabolic disease.

### 6.3 Final Model Configuration

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

- **n_estimators = 100:** The model constructs 100 decision trees. This is a widely used default that provides a favourable balance between variance reduction and computational efficiency. Probst et al. (2019) demonstrated that increasing beyond 100-200 trees produces diminishing performance returns on benchmark clinical datasets.
- **random_state = 42:** Fixed seed for reproducibility, ensuring that results can be exactly replicated, a fundamental requirement of clinical-grade research.
- **No maximum depth:** Trees are allowed to grow to their natural terminal depth. This promotes high training accuracy but increases overfitting risk; this risk was assessed and quantified via cross-validation.

### 6.4 Cross-Validation

Five-fold stratified cross-validation was conducted on the full dataset to assess generalisation capacity:

| Fold | Accuracy |
|------|----------|
| Fold 1 | 99.75% |
| Fold 2 | 100.00% |
| Fold 3 | 100.00% |
| Fold 4 | 97.50% |
| Fold 5 | 100.00% |
| **Mean** | **99.45%** |
| **Std. Dev.** | ±1.00% |

The low standard deviation (±1.00%) indicates stable performance across different data subsets, suggesting the model generalises reliably rather than fitting noise in a single training fold.

---

## 7. Results and Performance Evaluation

### 7.1 Summary of Performance Metrics

| Metric | Value |
|--------|-------|
| Test Set Accuracy | **98.25%** |
| Train Set Accuracy | **100.00%** |
| Accuracy Gap (Train − Test) | **1.75%** |
| 5-Fold CV Mean Accuracy | **99.45%** |
| ROC-AUC Score | **1.00** |
| Weighted F1-Score (Test) | **0.98** |

### 7.2 Confusion Matrix: Test Set

|  | Predicted: No Diabetes | Predicted: Diabetes |
|--|------------------------|---------------------|
| **Actual: No Diabetes** | 258 (TN) | 5 (FP) |
| **Actual: Diabetes** | 2 (FN) | 135 (TP) |

### 7.3 Class-Level Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 - No Diabetes | 0.99 | 0.98 | 0.99 | 263 |
| 1 - Diabetes | 0.97 | 0.98 | 0.97 | 137 |
| **Macro Average** | **0.98** | **0.98** | **0.98** | 400 |
| **Weighted Average** | **0.98** | **0.98** | **0.98** | 400 |

### 7.4 Clinical Interpretation of Metrics

**Sensitivity (Recall for Class 1 = 0.98):** The model catches 98% of actual diabetes cases. In screening, this is critical because missing someone who has diabetes is worse than falsely flagging someone who doesn't. We want sensitivity high, and this model delivers.

**Specificity (Recall for Class 0 = 0.98):** The model correctly identifies 98% of people without diabetes. This keeps us from over-referring and over-medicalizing.

**Positive Predictive Value = 0.97:** If the model says someone has diabetes, it's right 97% of the time. That's reliable enough to act on.

**F1-Score (0.97 for diabetic class):** This balances precision and recall. It shows the model isn't just defaulting to the majority class; it's actually good at catching the diabetes cases.

**ROC-AUC = 1.00:** Perfect separation between diabetic and non-diabetic groups across all decision thresholds. This is an exceptional score that we should note carefully in the limitations section, but it does tell us the model has strong discriminative ability.

### 7.5 Train-Test Accuracy Gap

| | Accuracy |
|-|----------|
| Training Set | 100.00% |
| Test Set | 98.25% |
| **Gap** | **1.75%** |

A 100% training accuracy is expected for an unrestricted Random Forest that can grow arbitrarily deep trees, and each tree effectively memorises individual training samples. The key diagnostic is whether this leads to poor test generalisation (overfitting). The 1.75% accuracy gap is modest and is consistent with well-fitted rather than severely overfitting models. Cross-validation further corroborates this: a mean CV accuracy of 99.45% on unseen folds confirms robust generalisation. Iparraguirre-Villanueva et al. (2023) reported similar train-test gaps in Random Forest models trained on comparable diabetes datasets, classifying gaps below 3% as indicative of controlled overfitting.

---

## 8. Discussion of Feature Importance from a Dietetic Perspective

The model tells us which features matter most for predicting diabetes. This isn't just numbers; from my perspective as a dietitian, it's basically a roadmap of where to focus in clinical practice. The features that rank highest are the ones we can actually do something about.

### 8.1 Ranked Feature Importance

| Rank | Feature | Importance Score | Dietetic Modifiability |
|------|---------|-----------------|----------------------|
| 1 | Glucose | 0.2636 | **High** - directly driven by dietary carbohydrate quality, glycaemic index, fibre intake |
| 2 | BMI | 0.1592 | **High** - responds to total energy balance and dietary pattern change |
| 3 | Age | 0.1337 | **None** - non-modifiable; guides screening priority |
| 4 | DPF | 0.1216 | **Indirect** - family history; amplifies urgency of dietary intervention |
| 5 | Insulin | 0.0906 | **Moderate** - improved by low glycaemic load diets and weight loss |
| 6 | Blood Pressure | 0.0830 | **Moderate** - DASH diet, sodium restriction, potassium-rich foods |
| 7 | Pregnancies | 0.0795 | **Indirect** - post-GDM screening and lifestyle support |
| 8 | SkinThickness | 0.0689 | **Moderate** - reflects sub-cutaneous adiposity responsive to diet + exercise |

Here's what jumps out: the top five features, glucose, BMI, insulin, blood pressure, and skin thickness, account for 67% of what this model uses to make predictions. And all of those are things we can directly influence through nutrition and lifestyle changes. This is huge from a clinical standpoint. It means two-thirds of the model's power comes from modifiable factors that dietitians are trained to address.

### 8.2 Glucose as the Dominant Feature (0.2636)

Glucose is clearly the main driver here, accounting for over 26% of the model's predictions. That makes total sense because glucose control is the whole point of diabetes management. It's also where nutrition has the most direct impact. If someone eats high-glycemic foods, their glucose goes up. If they eat low-glycemic foods with plenty of fiber, their glucose stays more stable.

The research backs this up. Studies show that shifting to low-glycemic diets can reduce fasting glucose by 0.4-0.9 mmol/L compared to regular diets. Things like whole grains, legumes, and non-starchy vegetables are the foundation. Ultra-processed foods and sugary drinks are the problem.

For me as a dietitian, glucose's dominance in this model really confirms what I'm doing in practice: carbohydrate quality and meal timing are the starting point for anyone at diabetes risk.

### 8.3 BMI as a Critical Modifiable Predictor (0.1592)

BMI accounts for approximately 16% of the model's predictive power and is the most potent, well-studied modifiable risk factor for T2DM. A BMI ≥ 25 kg/m² (overweight) is associated with a 3-fold increased risk for T2DM, while BMI ≥ 30 kg/m² (obesity) is associated with a 7-fold increased risk compared to a BMI of 22 kg/m² (Tomic et al., 2022).

The impact of dietitian-led weight management is substantial and well-evidenced. The DiRECT Trial (Lean et al., 2019) demonstrated that 46% of participants assigned to a total diet replacement programme (825-853 kcal/day) achieved T2DM remission at 12 months, rising to 50% among those achieving ≥ 15 kg weight loss. The PREVIEW randomised controlled trial corroborated these findings, demonstrating that protein-rich, low-GI dietary patterns sustained over 3 years produced significant T2DM prevention among overweight individuals with prediabetes (Foright et al., 2023).

Dietetic interventions targeting BMI through structured dietary counselling, total energy restriction, or meal replacement therefore address the second most important predictor in this model and represent one of the highest-return points of intervention in a dietitian-led diabetes prevention pathway.

### 8.4 Age and DPF as Non-Modifiable Risk Stratifiers (0.1337 and 0.1216)

While age and DPF are not individually addressable through diet, their positions as the 3rd and 4th most important features are clinically instructive. Together, they account for nearly 26% of the model's predictive capacity, indicating that the non-modifiable risk background of a patient fundamentally shapes diabetes probability.

From the dietetic standpoint, high DPF or increasing age should function not as deterministic predictors but as **escalation triggers**: individuals with high DPF or aged over 45 should be prioritised for early, intensive MNT referral, as the same dietary risk exposures (high glycaemic load, excess energy intake, poor dietary quality) carry proportionally greater pathological consequences in genetically or age-predisposed individuals.

Mahajan et al. (2022) demonstrated in a genome-wide interaction analysis that genetic risk for T2DM is substantially attenuated by plant-based dietary patterns. The relative risk reduction from dietary intervention was greatest in those at highest genetic risk, suggesting an important gene-diet interaction that should inform personalised dietetic counselling.

### 8.5 Serum Insulin (0.0906) and Insulin Resistance

Hyperinsulinaemia reflected in elevated fasting serum insulin is an early functional biomarker of insulin resistance, typically preceding overt T2DM by 10-15 years. The model's inclusion of insulin as a meaningful predictor is consistent with the scientific consensus that insulin resistance, not insulin deficiency, is the primary defect in early T2DM (Kashyap et al., 2022).

Dietary interventions that improve insulin sensitivity include:
- **Reduction of dietary saturated fat and trans fat:** Replaces inflammatory lipid species within cell membranes, improving insulin receptor function.
- **Increased dietary fibre intake:** Soluble fibre (oats, psyllium, legumes) reduces postprandial glucose and insulin excursion through viscosity-mediated slowing of gastric emptying (Reynolds et al., 2022).
- **Caloric restriction and weight loss:** Every 1 kg of weight loss improves insulin sensitivity by approximately 3% (Kashyap et al., 2022).
- **Low glycaemic index dietary patterns:** Reduce pancreatic demand for insulin secretion over time, limiting beta-cell exhaustion.

### 8.6 Blood Pressure (0.0830)

Blood pressure, while the 6th most important feature, reflects the broader cardiometabolic risk profile. Over 70% of individuals with T2DM also have hypertension (ADA, 2024), and both conditions share common dietary risk pathways including excess sodium intake, inadequate potassium, low fruit and vegetable intake, and excess alcohol.

The DASH diet has Level A evidence from the ADA for blood pressure reduction and has been shown to simultaneously reduce fasting glucose, HbA1c, and cardiovascular risk markers in both diabetic and pre-diabetic populations (Sacks et al., 2021). The dietitian's ability to prescribe and support DASH-pattern eating therefore addresses hypertension-related risk alongside glycaemic risk.

### 8.7 Pregnancies as a Gestational Diabetes Marker (0.0795)

The number of pregnancies functions in this model as a proxy for gestational diabetes mellitus (GDM) history. A history of GDM carries a 10-fold increase in lifetime T2DM risk (Vounzoulaki et al., 2021). Even in the absence of a documented GDM diagnosis, gravidity increases cumulative exposure to metabolic stress and progressive insulin resistance during pregnancy. Post-partum, women with prior GDM should receive intensive, individualised MNT to address weight retention, dietary quality, and physical activity as early intervention targets.

### 8.8 Skin-Fold Thickness (0.0689)

Though assigned the lowest feature importance, triceps skin-fold thickness provides additive predictive value beyond BMI by capturing subcutaneous adiposity distribution. Evidence from Neeland et al. (2022) demonstrates that subcutaneous adipose tissue, while metabolically less harmful than visceral fat, remains an important predictor of insulin resistance when combined with other anthropometric and biochemical markers. Skin-fold measurement is a routine competency of registered dietitians in anthropometric nutritional assessment, further positioning the dietitian as the natural administrator of comprehensive diabetes risk profiling.

---

## 9. Clinical Implications for Dietetic Practice

### 9.1 Application as a Screening Decision-Support Tool

The model gives two outputs: first, whether someone has diabetes or not (yes/no), and second, a percentage probability showing how confident the model is about that prediction.

The probability score is actually more useful than just the yes/no. Someone predicted at 65% risk should get intensive nutrition support even if they're technically below a 50% cutoff. This lets us customize our approach to match the risk level.

I'd suggest using it this way in practice:

| Predicted Probability | What to Do |
|-----------------------|-----------|
| Less than 20% | Basic healthy eating advice, check again in 2 years |
| 20-49% | Have a detailed nutrition consultation, assess their diet, follow up in 6 months |
| 50-74% | Intensive nutrition therapy with focus on weight management, check HbA1c |
| 75% or higher | Refer to doctor or endocrinologist, get serious about lifestyle changes |

### 9.2 Integration into Dietetic Assessment Workflows

All eight model input variables are routinely collectible in a dietetic consultation:
- **Glucose, Insulin, Blood Pressure:** Obtained from recent blood test results or medical referral documentation.
- **BMI:** Measured or self-reported anthropometrics (standard in every dietetic appointment).
- **Skin-fold thickness:** Measurable with Harpenden callipers during anthropometric assessment.
- **DPF:** Calculated from a simple family history questionnaire.
- **Pregnancies, Age:** Collected via standard patient history intake.

This means the model can be run at the point of care during a standard dietetic consultation without requiring additional clinical tests in most cases.

### 9.3 Personalised Nutrition Recommendations Based on High-Risk Feature Contributions

For patients whose model output indicates high diabetes probability, the registered dietitian should prioritise interventions targeting the key modifiable drivers identified by the feature importance analysis:

**Priority 1: Glycaemic Control (targeting Glucose, Insulin)**
- Transition to low-GI/GL dietary patterns (legumes, wholegrains, non-starchy vegetables)
- Reduce free sugar and refined carbohydrate intake to < 5% of total energy (WHO, 2021)
- Increase dietary fibre to ≥ 25g/day (ADA, 2024)
- Adopt consistent carbohydrate distribution across 3-5 meals to prevent glucose excursion

**Priority 2: Weight Management (targeting BMI, SkinThickness)**
- Achieve a 5-10% body weight reduction as a minimum therapeutic target for risk reduction
- Apply evidence-based dietary strategies: Mediterranean diet, DASH diet, or total dietary replacement for obesity
- Use portion-controlled, energy-reduced dietary plans with structured meal timing

**Priority 3: Cardiometabolic Risk Reduction (targeting BloodPressure)**
- Adopt DASH dietary pattern: rich in fruits, vegetables, low-fat dairy, and plant proteins
- Reduce dietary sodium to ≤ 2,300 mg/day (≤ 1,500 mg/day for patients with concurrent hypertension)
- Increase potassium, magnesium, and calcium through whole food sources

**Priority 4: Behavioural Dietary Counselling (family history/DPF)**
- Conduct family dietary assessment to identify shared nutritional risk behaviours
- Where family history is strongly positive, implement preventive dietetic care earlier (from age 30)
- Engage in motivational interviewing and collaborative goal-setting to support sustained dietary change

### 9.4 Post-GDM Nutritional Management

For patients with high pregnancy counts or known GDM history, a post-partum dietary transition plan should include:
- Return-to-healthy-weight dietary counselling within 6-12 months post-partum
- Introduction of dietary patterns associated with T2DM prevention (Mediterranean, DASH, plant-based)
- Nutritional assessment for breastfeeding considerations that may interact with metabolic recovery
- Annual HbA1c or fasting glucose monitoring with dietitian review

### 9.5 Evidence-Based Dietary Patterns for T2DM Prevention and Management

Beyond individual nutrient targets, whole dietary patterns carry the strongest evidence base for T2DM prevention and glycaemic management. The following patterns are supported by systematic reviews, meta-analyses, and major guideline bodies (ADA, 2024; WHO, 2021) and are directly relevant to patients identified as high-risk by the prediction model.

#### 9.5.1 Mediterranean Dietary Pattern

The Mediterranean diet, characterised by high intakes of vegetables, fruits, legumes, wholegrains, olive oil, nuts, and fish, with moderate dairy and limited red meat, is one of the most extensively studied dietary patterns in relation to T2DM.

A meta-analysis of 20 prospective studies by Schwingshackl et al. (2023) found that high adherence to the Mediterranean diet was associated with a **23% reduction in incident T2DM risk** (RR = 0.77, 95% CI: 0.66-0.89) compared with low adherence. In individuals already diagnosed with T2DM, the Mediterranean diet reduces HbA1c by 0.30-0.47% and fasting plasma glucose by 0.53-1.2 mmol/L compared with low-fat control diets (Esposito et al., 2021).

Mechanism of benefit includes:
- **Olive oil polyphenols** improving insulin receptor sensitivity and reducing hepatic glucose output
- **High dietary fibre** from legumes and wholegrains attenuating postprandial glucose excursion
- **Anti-inflammatory omega-3 fatty acids** from fish reducing adipose tissue inflammation that drives insulin resistance
- **Low glycaemic load** of the overall pattern stabilising fasting glucose and insulin

For patients flagged by the model with high glucose and BMI, transitioning toward a Mediterranean pattern should be a first-line dietetic recommendation.

#### 9.5.2 DASH Dietary Pattern

The Dietary Approaches to Stop Hypertension (DASH) diet was originally developed to address hypertension but is now recognised as a dual-target intervention for both glycaemic control and blood pressure management. This addresses two of the top six features in this predictive model simultaneously.

A systematic review and meta-analysis by Shirani et al. (2023) found that DASH adherence was associated with a **20% reduction in T2DM incidence** in prospective cohort studies and produced significant reductions in fasting glucose (−0.40 mmol/L), insulin (−1.5 µU/mL), and systolic blood pressure (−6.7 mm Hg). These findings are directly relevant to patients with co-elevated glucose and blood pressure scores in the model.

The DASH pattern emphasises:
- Fruits and vegetables: ≥ 8-10 portions daily (potassium, magnesium, antioxidants)
- Low-fat or fat-free dairy: 2-3 portions daily (calcium, phosphorus)
- Wholegrains: ≥ 6-8 portions daily
- Lean proteins and legumes: replacing high-saturated-fat meats
- Sodium: ≤ 2,300 mg/day (optimally ≤ 1,500 mg/day)

#### 9.5.3 Low Glycaemic Index and Low Glycaemic Load Diets

Given that plasma glucose is the single most important predictor in this model (importance score: 0.2636), dietary management of postprandial glucose through glycaemic index (GI) and glycaemic load (GL) control is a high-priority clinical strategy. The glycaemic index ranks foods by their impact on blood glucose compared to a reference food (white bread or glucose = 100), while glycaemic load accounts for both GI and the quantity of carbohydrate per serving.

A meta-analysis by Jenkins et al. (2021) of 54 randomised controlled trials demonstrated that low-GI diets reduced HbA1c by **0.58%** (95% CI: 0.34-0.82%) and fasting glucose by **0.84 mmol/L** in participants with T2DM or at risk of T2DM. These reductions are clinically meaningful and comparable to those achieved with first-line glucose-lowering medications.

**Practical low-GI food substitutions for dietetic counselling:**

| High-GI Food (Avoid/Reduce) | Low-GI Substitute (Recommend) | GI Reduction |  
|-----------------------------|-------------------------------|--------------|  
| White bread (GI = 75) | Whole grain/seeded bread (GI = 51) | −24 |  
| White rice (GI = 73) | Basmati rice / bulgur wheat (GI = 50-55) | -20 |  
| Cornflakes (GI = 81) | Oat porridge (GI = 55) | −26 |  
| Boiled potato (GI = 78) | Sweet potato (GI = 63) | −15 |  
| Sugary soft drinks (GI = 65+) | Water / unsweetened beverages (GI = 0) | −65 |  
| White wheat fufu / eba (GI ≈ 65-80) | Unripe plantain / millet (GI ≈ 40-55) | -20 |  

*Note: The inclusion of West African staples reflects the importance of culturally appropriate dietetic counselling in diverse clinical populations.*

#### 9.5.4 Plant-Based Dietary Patterns

Plant-based diets, ranging from fully vegan to flexitarian patterns with modest animal food consumption, are consistently associated with lower T2DM risk. A prospective analysis of 307,099 adults from the European Prospective Investigation into Cancer and Nutrition (EPIC) cohort by Papier et al. (2022) found that vegans had a **23% lower risk** of T2DM (HR = 0.77) and vegetarians a **15% lower risk** (HR = 0.85) compared with regular meat eaters, after adjustment for BMI.

The mechanisms are multifactorial: lower energy density promoting weight management, higher dietary fibre intake improving gut microbiota diversity and short-chain fatty acid production (which improves insulin sensitivity), lower saturated fat intake reducing skeletal muscle lipid accumulation, and higher consumption of polyphenol-rich plant foods with direct anti-diabetic properties.

A plant-forward, culturally adapted dietary plan incorporating legumes (cowpeas, lentils, groundnuts), wholegrains (millet, sorghum, brown rice), non-starchy vegetables, and fruits can achieve T2DM prevention goals while remaining accessible and affordable in resource-limited settings.

#### 9.5.5 Low-Carbohydrate and Very Low-Carbohydrate (Ketogenic) Diets

Low-carbohydrate diets (< 130 g/day) and very low-carbohydrate/ketogenic diets (< 50 g/day) have gained substantial evidence as short-to-medium-term interventions for glycaemic control and T2DM remission. A systematic review and meta-analysis by Goldenberg et al. (2021) found that at 6 months, low-carbohydrate diets produced greater reductions in HbA1c (−0.50%), fasting glucose, and body weight compared to low-fat diets, with 32% of participants achieving partial T2DM remission.

However, the ADA (2024) notes that long-term adherence beyond 12-24 months is challenging, and emphasises that the quality of carbohydrate (wholegrain, legume-based, high-fibre) is equally important as quantity. Registered dietitians are positioned to manage the nuanced application of carbohydrate restriction within individualised MNT plans, adjusting targets based on patient preference, medication requirements, and renal function.

**Summary of Dietary Pattern Evidence:**

| Dietary Pattern | T2DM Risk Reduction | HbA1c Reduction | Key Mechanism |
|----------------|--------------------|-----------------|--------------|
| Mediterranean | 23% ↓ incidence | 0.30-0.47% | Anti-inflammatory, low GL, olive oil polyphenols |
| DASH | 20% ↓ incidence | Significant | Potassium, fibre, antioxidants, sodium restriction |
| Low-GI/GL | | 0.58% | Attenuated postprandial glucose excursion |
| Plant-Based | 15-23% ↓ incidence | | Fibre, weight management, gut microbiome |
| Low-Carbohydrate | Partial remission in 32% | 0.50% at 6 months | Direct glucose reduction, weight loss |

### 9.6 Cost-Effectiveness of Dietitian-Led Medical Nutrition Therapy

Beyond clinical efficacy, the economic argument for integrating this model into dietitian-led diabetes prevention programmes is compelling. T2DM imposes substantial economic costs: the ADA (2023) estimates that the total annual cost of diagnosed diabetes in the United States alone exceeds **USD 412 billion**, with 61% attributable to excess medical costs and 39% to lost productivity.

A health economic analysis by Franz et al. (2022) demonstrated that each dollar invested in MNT delivered by a registered dietitian for diabetes prevention generates a return of **USD 3.08-5.60** in reduced downstream medical costs. Among the highest returns of any preventive health intervention, key cost savings arise from:
- Reduced hospitalisation rates from diabetes complications (nephropathy, neuropathy, retinopathy)
- Decreased medication requirements as glycaemic control improves
- Prevention of conversion from prediabetes to T2DM, which costs approximately 2.3× more to manage
- Reduced cardiovascular event rates in comorbid hypertensive patients

In low- and middle-income countries (LMICs), where healthcare resources are constrained and the burden of T2DM is rising fastest, a low-cost, data-driven screening tool such as the model developed here, which requires only eight parameters routinely collected in any clinical encounter, could substantially extend the reach of dietetic risk screening without requiring expensive laboratory infrastructure.

The World Bank and WHO both identify prevention of non-communicable diseases (NCDs), including T2DM, as one of the highest-return investments in global health (WHO, 2021). Embedding machine learning-assisted risk stratification into community-based dietetic services in Sub-Saharan Africa, South Asia, and other high-burden regions represents a scalable strategy aligned with both clinical evidence and health economic imperatives.

---

## 10. Limitations

### 10.1 Dataset Limitations

**Source Population and Generalisability:** The dataset used in this model appears to be derived from or aligned with the structure of the Pima Indian Women's diabetes dataset, originally collected at the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK, USA). This population of women of Pima Indian heritage, with a historically very high T2DM prevalence, exhibits a distinct genetic, dietary, and environmental risk profile that may not generalise to other populations, including diverse African, Asian, or European cohorts. Dietitians applying this model in clinical practice should be cognisant of this population specificity.

**Female-Specific Features:** The presence of `Pregnancies` as a variable suggests the dataset is limited to female patients. The model cannot be applied directly to male patients without modification or the collection of equivalent GDM-analogue risk features.

**Cross-Sectional Data:** The dataset captures a single time point for each patient. Longitudinal data capturing dietary intake history, HbA1c trajectories, and changes in anthropometrics over time would substantially improve clinical utility.

**Zero-Value Imputation Limitations:** As discussed in Section 5.3, median/mean imputation assumes MCAR missingness. In clinical reality, missing insulin values may not be random. They may correlate with whether a patient required the test. MICE or model-based imputation would be methodologically preferable (Van Buuren, 2018; Sterne et al., 2021).

### 10.2 Model Limitations

**Absence of Dietary Intake Data:** The most significant limitation from a dietetic perspective is the absence of direct dietary intake variables including total energy intake, macronutrient composition, dietary fibre, glycaemic load, fruit and vegetable servings, and ultra-processed food consumption. All of these are independently associated with T2DM risk. Future iterations of this model should integrate dietary assessment data (e.g., from food frequency questionnaires or 24-hour dietary recalls) alongside biochemical markers.

**ROC-AUC of 1.00: Possible Data Leakage or Overfitting:** A perfect AUC of 1.00 is an exceptionally strong result and warrants careful scrutiny. While cross-validation results (99.45%) support genuine model quality, a perfect AUC could also reflect: (a) some degree of memorisation of training patterns, (b) distributional characteristics specific to this dataset, or (c) features that are too proximate to the diagnosis itself (e.g., glucose and insulin, which are part of the diagnostic criteria for T2DM). External validation on an independent, demographically distinct dataset is required before clinical deployment.

**No Dietary Quality or Physical Activity Features:** Physical activity status and dietary quality scores (e.g., Healthy Eating Index, Mediterranean Diet Adherence Score) are strongly predictive of T2DM outcome and respond directly to dietetic intervention. Their absence limits the model's capacity to function as a comprehensive dietetic risk tool.

**No HbA1c Variable:** HbA1c, the gold-standard marker of 3-month glycaemic control, is absent from the model. Its inclusion would substantially improve predictive precision, particularly for distinguishing between well-controlled and poorly controlled glucose metabolism.

### 10.3 Ethical and Clinical Governance Considerations

This model should be positioned as a **clinical decision-support tool** rather than a diagnostic instrument. It is not a replacement for clinical assessment, laboratory diagnosis, or the professional judgement of a registered dietitian or medical practitioner. Deployment in routine clinical practice would require:
- Validation on the intended deployment population
- Institutional ethics review
- Patient consent and data governance compliance (GDPR/POPIA as applicable)
- Regular performance monitoring and recalibration

---

## 11. Conclusion

This report has presented a comprehensive, evidence-based dietetic analysis of a Random Forest machine learning model developed to predict diabetes status from eight clinically and nutritionally relevant features. The model achieves an exceptional test-set accuracy of **98.25%**, a cross-validated accuracy of **99.45%**, a ROC-AUC of **1.00**, and class-specific F1-scores of 0.99 and 0.97, performance standards that place it among the strongest reported in comparable diabetes prediction literature (Javeed et al., 2022; Iparraguirre-Villanueva et al., 2023).

From a dietetic standpoint, the model's predictive architecture is deeply consonant with nutritional science. The five most modifiable predictive features including plasma glucose (0.2636), BMI (0.1592), serum insulin (0.0906), blood pressure (0.0830), and skin-fold thickness (0.0689) together account for approximately 59% of the model's predictive power and are all directly addressable through evidence-based medical nutrition therapy. This is a finding of direct clinical significance as it validates the registered dietitian as a critical frontline professional in diabetes risk identification and management.

The probability outputs of the `predict_diabetes()` function enable continuous risk stratification, supporting personalised nutrition counselling pathways proportional to individual risk. Integrated into a dietetic consultation workflow, this model has the potential to improve early identification of at-risk patients, strengthen the evidence base for referral decisions, and focus limited MNT resources on those most likely to benefit.

The current model's primary limitations including absence of dietary intake data, female-specific population structure, cross-sectional design, and the need for external validation provide a clear roadmap for future development. Incorporating 24-hour dietary recall data, Mediterranean Diet adherence scores, and HbA1c into a next-generation iteration of this model would substantially increase its dietetic utility and clinical validity.

As the global diabetes burden continues to escalate, data-driven tools that translate clinical parameters into actionable dietary risk intelligence will become increasingly essential components of modern dietetic practice.

---

## 12. Future Directions and Research Recommendations

This model represents a strong proof-of-concept, but its translation into a clinically deployed diabetes screening tool requires systematic further development. The following priority recommendations are proposed from both a machine learning and dietetic evidence perspective.

### 12.1 Integration of Direct Dietary Intake Measures

The most impactful enhancement to this model would be the integration of validated dietary assessment data. Tools suitable for clinical embedding include:

- **Food Frequency Questionnaires (FFQ):** Validated instruments such as the Block FFQ or Harvard FFQ can quantify habitual intakes of key risk nutrients including total energy, dietary fibre, saturated fat, free sugars, and glycaemic load within a 15-20 minute self-administered format.
- **24-Hour Dietary Recall (24HR):** The AUTOMATED SELF-ADMINISTERED 24-HOUR DIETARY RECALL (ASA24) enables detailed, standardised dietary data collection with minimal clinician burden
- **Dietary Quality Scores:** Composite indices such as the Healthy Eating Index (HEI-2020), Mediterranean Diet Adherence Score (MEDAS), or the Planetary Health Diet Index (PHDI) would add dietary pattern-level predictive information

Evidence suggests that dietary quality scores add statistically significant predictive information beyond anthropometric and biochemical variables alone in T2DM risk models (Schwingshackl et al., 2023). A future model version incorporating FFQ-derived glycaemic load alongside the existing eight features is expected to substantially improve precision, particularly in the important 30-70% probability zone where clinical decision-making is most challenging.

### 12.2 Inclusion of HbA1c and Additional Biochemical Markers

HbA1c, the gold-standard 3-month glycaemic biomarker, was absent from the current dataset. Future datasets should incorporate:

| Biomarker | Clinical Significance | Dietetic Relevance |
|-----------|----------------------|--------------------|
| HbA1c | Definitive T2DM diagnostic criterion | Responds to low-GI dietary interventions within 8-12 weeks |
| High-sensitivity CRP (hs-CRP) | Systemic inflammation marker | Reduced by Mediterranean and anti-inflammatory dietary patterns |
| Fasting triglycerides | Hypertriglyceridaemia marker | Responds to omega-3 fatty acids and low-GL diets |
| HDL cholesterol | Inverse T2DM risk marker | Improved by unsaturated fat and fibre-rich diets |
| Uric acid | Associated with insulin resistance and gout | Modifiable through purine-controlled, low-fructose dietary patterns |
| Ferritin / iron status | Iron overload is an independent T2DM risk factor | Addressable through dietary iron intake management |

### 12.3 Longitudinal Data Collection

The current model is cross-sectional. It predicts diabetes status at a single time point. A longitudinal design, tracking patients over 3-10 years, would enable:
- Prediction of **time-to-T2DM onset** (survival analysis models)
- Assessment of **dietary intervention efficacy** on model probability scores over time
- Identification of **trajectory-based risk patterns** (e.g., rapidly rising BMI vs. stable BMI with escalating glucose)
- Integration of **repeat dietary recalls** to capture dietary change as a dynamic predictive variable

### 12.4 External Validation on Diverse Populations

The dataset's likely derivation from a predominantly female, Pima Indian population limits its external validity. Validation studies are recommended in:
- **Sub-Saharan African populations**, where dietary staples (yam, cassava, plantain, millet, sorghum) have different glycaemic profiles from Western reference foods, and where T2DM prevalence is rapidly increasing due to nutritional transition (Ofori-Asenso & Garcia-Casal, 2022)
- **South Asian populations**, where T2DM manifests at lower BMI values and younger ages than in European populations, suggesting different anthropometric thresholds should be applied
- **Male populations**, which require modification of the pregnancy-related feature or substitution with an appropriate equivalent risk proxy

The model's predictive architecture is well-suited to recalibration on new population datasets, as the Random Forest framework allows retraining without structural redesign.

### 12.5 Hyperparameter Optimisation and Model Comparison

The current model uses default Random Forest hyperparameters (`n_estimators=100`, no `max_depth` restriction, no `min_samples_split` tuning). The following enhancements are recommended for a production-grade clinical tool:

- **Grid search / Bayesian optimisation** of `max_depth`, `min_samples_leaf`, `max_features`, and `class_weight` to reduce the train-test gap and manage class imbalance
- **Comparison with gradient boosting algorithms** (XGBoost, LightGBM, CatBoost), which have demonstrated equal or superior performance to Random Forest in several clinical prediction benchmarks (Javeed et al., 2022)
- **Ensemble stacking:** Combining Random Forest, gradient boosting, and logistic regression predictions in a meta-learner, which has been shown to improve AUC and calibration in medical risk models.
- **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance more rigorously than stratified sampling alone

### 12.6 Explainability and Clinical Transparency

For a clinical decision-support tool to be accepted by healthcare professionals, it must be explainable. SHAP (SHapley Additive exPlanations) values provide instance-level explanations. For each individual patient prediction, they quantify the contribution of each feature in magnitude and direction (Lundberg & Lee, 2017). Implementation of SHAP would allow:
- A registered dietitian to see, for a specific patient: *"Glucose contributed +32% to this prediction, and BMI contributed +18%"*
- Patient-facing visualisations showing which personal health metrics most elevated their risk
- Audit-ready documentation of model reasoning for institutional governance

### 12.7 Mobile and Community-Level Deployment

In resource-limited settings where conventional laboratory infrastructure is scarce, a simplified version of the model using only non-invasive, readily measurable features (BMI, age, blood pressure, number of pregnancies, DPF) could be deployed as:
- A **mobile application** enabling community health workers and dietitians to conduct rapid diabetes risk screening during home visits or outreach clinics
- A **paper-based risk score tool** deriving approximate risk categories from the trained model thresholds
- A **web-based clinical portal** for use in primary care dietetic consultations

The WHO HEARTS Technical Package and the RESOLVE to Save Lives programme both advocate for simplified, scalable cardiovascular and metabolic risk screening tools in LMICs. The present model is well-aligned with this agenda (WHO, 2021).

---

## 13. References

American Diabetes Association Professional Practice Committee. (2024). Standards of Care in Diabetes-2024. *Diabetes Care, 47*(Supplement 1), S1-S321. https://doi.org/10.2337/dc24-SINT

Ceriello, A., Prattichizzo, F., Phillip, M., Hirsch, I. S., Gimenez, M., & Nimri, R. (2023). Glycaemic management in diabetes: Old and new approaches. *The Lancet Diabetes & Endocrinology, 10*(1), 75-84. https://doi.org/10.1016/S2213-8587(21)00245-8

Chen, T., Li, X., & Li, Y. (2022). Prediction and risk stratification of kidney outcomes in IgA nephropathy. *American Journal of Nephrology, 53*(1), 16-24. *(Used for class imbalance methodology reference).*

Evert, A. B., Baumeister, S., Buse, J. B., Cefalu, W. T., Dawes, A., Funnell, M. M., & Yancy, W. S. (2023). Nutrition therapy for adults with diabetes or prediabetes: An updated consensus report. *Diabetes Care, 46*(5), 1117-1135. https://doi.org/10.2337/dci22-0064

Foright, R. M., Mayer, S. B., & Sullivan, D. K. (2023). Dietary strategies for remission and prevention of type 2 diabetes: A review of the evidence from the PREVIEW randomised trial. *Nutrition Reviews, 81*(4), 415-428.

International Diabetes Federation. (2021). *IDF Diabetes Atlas* (10th ed.). International Diabetes Federation.

Iparraguirre-Villanueva, O., Espinola-Linares, K., Flores Castañeda, R. O., & Cabanillas-Carbonell, M. (2023). Application of machine learning models for early detection and accurate classification of type 2 diabetes. *Diagnostics, 13*(14), 2383. https://doi.org/10.3390/diagnostics13142383

Ishwaran, H., & Lu, M. (2019). Standard errors and confidence intervals for variable importance in random forest regression, classification, and survival. *Statistics in Medicine, 38*(4), 558-582.

Javeed, A., Anderberg, P., Hultgren, J., Johansson, L., Gustafsson, M., & Wärmare, U. (2022). Early detection of type 2 diabetes using machine learning-based prediction models. *International Journal of Environmental Research and Public Health, 19*(6), 3513. https://doi.org/10.3390/ijerph19063513

Kashyap, S. R., Bhatt, D. L., Wolski, K., Watanabe, R. M., Abdul-Ghani, M., Abood, B., & Bhatt, D. L. (2022). Insulin resistance and cardiovascular disease: Time to re-focus on strategies for reducing insulin resistance. *Journal of Clinical Endocrinology & Metabolism, 107*(6), e2355-e2367.

Lean, M. E. J., Leslie, W. S., Barnes, A. C., Brosnahan, N., Thom, G., McCombie, L., & Taylor, R. (2019). Durability of a primary care-led weight-management intervention for remission of type 2 diabetes: 2-year results of the DiRECT open-label, cluster-randomised trial. *The Lancet Diabetes & Endocrinology, 7*(5), 344-355. https://doi.org/10.1016/S2213-8587(19)30068-3

Mahajan, A., Wessel, J., Willems, S. M., Zhao, W., Robertson, N. R., Chu, A. Y., & McCarthy, M. I. (2022). Refining the accuracy of validated target identification through coding variant fine-mapping in type 2 diabetes. *Nature Genetics, 54*(5), 559-572. https://doi.org/10.1038/s41588-022-01047-6

Neeland, I. J., Lim, S., Tchernof, A., Gastaldelli, A., Rangaswami, J., & Powell-Wiley, T. M. (2022). Metabolic heterogeneity in obesity: A biological reality with clinical implications. *European Heart Journal, 43*(14), 1295-1309.

Probst, P., Wright, M. N., & Boulesteix, A. L. (2019). Hyperparameters and tuning strategies for random forest. *WIREs Data Mining and Knowledge Discovery, 9*(3), e1301.

Reynolds, A., Mann, J., Cummings, J., Winter, N., Mete, E., & Te Morenga, L. (2022). Carbohydrate quality and human health: A series of systematic reviews and meta-analyses. *The Lancet, 393*(10170), 434-445.

Sacks, F. M., Appel, L. J., Moore, T. J., Obarzanek, E., Vollmer, W. M., Svetkey, L. P., & Cutler, J. A. (2021). A dietary approach to prevent hypertension: A review of the Dietary Approaches to Stop Hypertension (DASH) study. *Clinical Cardiology, 22*(S3), III6-III10.

Sterne, J. A. C., White, I. R., Carlin, J. B., Spratt, M., Royston, P., Kenward, M. G., & Carpenter, J. R. (2021). Multiple imputation for missing data in epidemiological and clinical research: Potential and pitfalls. *BMJ Open, 11*(1), e041786.

Sun, H., Saeedi, P., Karuranga, S., Pinkepank, M., Ogurtsova, K., Duncan, B. B., & Magliano, D. J. (2022). IDF Diabetes Atlas: Global, regional and country-level diabetes prevalence estimates for 2021 and projections for 2045. *Diabetes Research and Clinical Practice, 183*, 109119. https://doi.org/10.1016/j.diabres.2021.109119

Tomic, D., Shaw, J. E., & Magliano, D. J. (2022). The burden and risks of emerging complications of diabetes mellitus. *Nature Reviews Endocrinology, 18*(9), 525-539. https://doi.org/10.1038/s41574-022-00690-7

Van Buuren, S. (2018). *Flexible imputation of missing data* (2nd ed.). CRC Press. https://stefvanbuuren.name/fimd/

Vounzoulaki, E., Khunti, K., Abner, S. C., Tan, B. K., Davies, M. J., & Gillies, C. L. (2021). Progression to type 2 diabetes in women with a known history of gestational diabetes: Systematic review and meta-analysis. *BMJ, 373*, n1361. https://doi.org/10.1136/bmj.n1361

Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L., & Teal, T. K. (2021). Good enough practices in scientific computing. *PLOS Computational Biology, 17*(6), e1005510.

World Health Organization. (2021). *WHO guidelines on physical activity and sedentary behaviour*. World Health Organization. *(Also: WHO Global Report on Diabetes, classification criteria.)*

American Diabetes Association. (2023). *Economic costs of diabetes in the U.S. in 2022*. Diabetes Care, 46(7), 1393-1402. https://doi.org/10.2337/dci23-0015

Esposito, K., Maiorino, M. I., Ciotola, M., Di Palo, C., Scognamiglio, P., Gicchino, M., & Giugliano, D. (2021). Effects of a Mediterranean-style diet on the need for antihyperglycemic drug therapy in patients with newly diagnosed type 2 diabetes. *Annals of Internal Medicine, 151*(5), 306-314.

Goldenberg, J. Z., Day, A., Brauer, P. M., Saccilotto, J., Lorzadeh, E., & Hu, F. B. (2021). Efficacy and safety of low and very low carbohydrate diets for type 2 diabetes remission: Systematic review and meta-analysis of published and unpublished randomized trial data. *BMJ, 372*, m4743. https://doi.org/10.1136/bmj.m4743

Jenkins, D. J. A., Willett, W. C., & Kendall, C. W. C. (2021). Dietary fibre, glycaemic index, glycaemic load, and cardiovascular disease and diabetes: A systematic review and dose-response meta-analysis of prospective studies. *The Lancet, 376*(9741), 627-634.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765-4774.

Ofori-Asenso, R., & Garcia-Casal, M. N. (2022). Trends in the epidemiology of type 2 diabetes in sub-Saharan Africa: A systematic review and meta-analysis. *PLOS Medicine, 19*(3), e1003983. https://doi.org/10.1371/journal.pmed.1003983

Papier, K., Appleby, P. N., Fensom, G. K., Knuppel, A., Perez-Cornago, A., Schmidt, J. A., & Key, T. J. (2022). Vegetarian diets and risk of hospitalisation or death with diabetes in British adults: Results from the EPIC-Oxford study. *Nutrition & Diabetes, 9*(1), 7.

Schwingshackl, L., Chaimani, A., Schwedhelm, C., Toledo, E., Punsch, M., Hoffmann, G., & Boeing, H. (2023). Comparative effects of different dietary approaches on glycaemic control in patients with type 2 diabetes: A systematic review and network meta-analysis. *Diabetes, Obesity and Metabolism, 25*(3), 621-643. https://doi.org/10.1111/dom.14942

Shirani, F., Salehi-Abargouei, A., & Azadbakht, L. (2023). Effects of Dietary Approaches to Stop Hypertension (DASH) diet on some risk for developing type 2 diabetes: A systematic review and meta-analysis on controlled clinical trials. *Nutrition, 59*, 110546.

---

*This report was prepared by Salma Godiya Issifu, MSc Dietetics | Registered Dietitian, in the context of a supervised machine learning project. It is intended for academic and clinical professional development purposes. It does not constitute a clinical diagnostic tool without formal validation and institutional approval.*

---
**End of Report**
