# 🩺 Diabetes Prediction using Support Vector Machines (SVM)

A machine learning project that predicts whether a patient is diabetic or non-diabetic based on diagnostic health metrics, using a Support Vector Machine classifier with a linear kernel.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Making a Prediction](#making-a-prediction)

---

## Overview

This project builds a binary classification model to detect diabetes using the **Pima Indians Diabetes Dataset**. The model uses a **Support Vector Machine (SVM)** with a linear kernel, combined with feature standardization, to classify patients as either diabetic (`1`) or non-diabetic (`0`).

---

## Dataset

**File:** `diabetesdata.csv`  
**Shape:** 768 rows × 9 columns  
**Source:** Pima Indians Diabetes Dataset (originally from the National Institute of Diabetes and Digestive and Kidney Diseases)

### Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (genetic influence score) |
| `Age` | Age in years |
| `Outcome` | Target variable — `1` (Diabetic), `0` (Non-Diabetic) |

### Class Distribution

| Class | Count |
|---|---|
| Non-Diabetic (0) | 500 |
| Diabetic (1) | 268 |

---

## Project Workflow

1. **Data Loading** — Load the CSV dataset using pandas
2. **Exploratory Data Analysis (EDA)** — Inspect shape, data types, missing values, and class distribution
3. **Feature & Label Separation** — Split into feature matrix `X` and target vector `y`
4. **Feature Scaling** — Apply `StandardScaler` to normalize all features
5. **Train-Test Split** — 80/20 split with `stratify=y` to maintain class balance (`random_state=20`)
6. **Model Training** — Fit an `SVC(kernel='linear')` on the training set
7. **Evaluation** — Measure accuracy on both training and test sets
8. **Prediction** — Run inference on a new patient data point

---

## Results

| Split | Accuracy |
|---|---|
| Training Set | ~77.4% |
| Test Set | ~80.5% |

The model generalizes well — test accuracy is actually slightly higher than training accuracy, indicating no overfitting.

---

## Technologies Used

- Python 3.11
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
  - `SVC` (Support Vector Classifier)
  - `StandardScaler`
  - `train_test_split`
  - `accuracy_score`

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-svm.git
   cd diabetes-prediction-svm
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Add the dataset**  
   Place `diabetesdata.csv` in the root directory of the project.

---

## Usage

Launch the Jupyter Notebook and run all cells in order:

```bash
jupyter notebook Diabetes_Prediction_using_Support_Vector_Machines.ipynb
```

---

## Making a Prediction

At the end of the notebook, you can provide a custom patient input to get a prediction:

```python
input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)
# Format: (Pregnancies, Glucose, BloodPressure, SkinThickness,
#          Insulin, BMI, DiabetesPedigreeFunction, Age)

input_array = np.asarray(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)
prediction = model.predict(scaled_input)

if prediction[0] == 1:
    print("Diabetic")
else:
    print("Non-Diabetic")
```

