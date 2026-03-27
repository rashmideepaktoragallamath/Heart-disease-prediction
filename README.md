# Heart Disease Prediction using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/drive/1JFpJMnTqiKG81__K5G9vqYXlDwVYJVdB?usp=sharing)
---

## 📋 **Project Overview**

Cardiovascular diseases (CVDs) are the leading cause of death globally. Early and accurate detection is critical for timely intervention and effective treatment. This project presents a comprehensive comparative analysis of **four supervised machine learning models** to predict the presence of heart disease in patients based on key clinical and demographic features.

### 🎯 **Objective**
To develop, evaluate, and compare multiple machine learning classification models to identify the most effective algorithm for predicting heart disease using the Cleveland Heart Disease dataset.

### 🤖 **Models Implemented**
| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | Ensemble | Creates multiple decision trees and averages their predictions |
| **K-Nearest Neighbors (KNN)** | Instance-based | Classifies based on K closest training instances |
| **Support Vector Machine (SVM)** | Margin-based | Finds optimal hyperplane to separate classes |
| **Decision Tree** | Tree-based | Creates flowchart-like structure of decisions |

---

## 📊 **Dataset**

### Source
- **Dataset:** Cleveland Heart Disease Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** 303 patient records
- **Features:** 13 clinical features + 1 target variable

### Features Description

| Feature | Description | Values |
|---------|-------------|--------|
| **age** | Age in years | 29-77 |
| **sex** | Gender | 1 = male, 0 = female |
| **cp** | Chest pain type | 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic |
| **trestbps** | Resting blood pressure | 94-200 mm Hg |
| **chol** | Serum cholesterol | 126-564 mg/dl |
| **fbs** | Fasting blood sugar | 1 = > 120 mg/dl, 0 = ≤ 120 mg/dl |
| **restecg** | Resting ECG results | 0 = normal, 1 = ST-T abnormality, 2 = probable/definite LVH |
| **thalach** | Maximum heart rate achieved | 71-202 bpm |
| **exang** | Exercise induced angina | 1 = yes, 0 = no |
| **oldpeak** | ST depression induced by exercise | 0.0-6.2 |
| **slope** | Slope of peak exercise ST segment | 0 = upsloping, 1 = flat, 2 = downsloping |
| **ca** | Number of major vessels colored | 0-4 |
| **thal** | Thalassemia | 1 = normal, 2 = fixed defect, 3 = reversible defect |
| **target** | Diagnosis | 0 = no heart disease, 1 = heart disease |

---

## 🛠️ **Methodology**

### 1. Data Preprocessing

```python
# Steps performed:
✅ Checked for missing values (none found)
✅ Validated data integrity (no invalid rows)
✅ One-Hot Encoding for categorical variables (cp, restecg, thal)
✅ Box-Cox transformation for skewed features (age, trestbps, chol, thalach, oldpeak)
✅ Feature scaling using StandardScaler for KNN and SVM
✅ Train-test split (80% training, 20% testing) with stratification
