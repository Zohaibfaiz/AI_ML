# 📈 Linear Regression — ML Model

<p align="center"> <img src="https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python" /> <img src="https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" /> </p>

<p align="center"> <b>A practical Linear Regression project using Python, Scikit-learn, and a real-world Student Performance dataset.</b> </p>

---
📌 Project Overview

This project demonstrates Linear Regression with two examples:

👨‍💼 Salary Prediction — a simple example using years of experience.
🎓 Student Exam Score Prediction — a real-world regression project using student performance data.

The main goal is to build a complete, practical regression project without unnecessary mathematical detail.



## 🧠 What is Linear Regression?

**Linear Regression** is one of the most fundamental supervised machine learning algorithms. It models the **linear relationship** between an independent variable (or multiple variables) and a continuous dependent variable.


## 🔁 Project Workflow

The notebook follows this step-by-step machine learning pipeline:

```
Import Libraries
      ↓
Load / Create Dataset
      ↓
Exploratory Data Analysis (EDA)
      ↓
Handle Missing Values & Duplicates
      ↓
Encode Categorical Variables
      ↓
Feature Scaling (StandardScaler)
      ↓
Train-Test Split (80% / 20%)
      ↓
Train LinearRegression Model
      ↓
Predict on Test Data
      ↓
Evaluate Performance (MAE, MSE, RMSE, R²)
      ↓
Visualize Results
```

---

## 💻 Implementation 1 — Simple Salary Prediction (Scratch Example)

### Dataset

A manually created synthetic dataset with perfect linear relationship:

| Years Experience | Salary (₹) |
|-----------------|------------|
| 1 | 30,000 |
| 2 | 40,000 |
| 3 | 50,000 |
| 4 | 60,000 |
| 5 | 70,000 |
| 6 | 80,000 |
| 7 | 90,000 |
| 8 | 1,00,000 |
| 9 | 1,10,000 |
| 10 | 1,20,000 |


### Results

| Metric | Value | Meaning |
|--------|-------|---------|
| **Slope (m)** | ~10,000 | Every 1 year of experience → +₹10,000 salary |
| **Intercept (b)** | ~20,000 | Base salary = ₹20,000 |
| **MAE** | 0.0 | Zero mean absolute error |
| **MSE** | 0.0 | Zero mean squared error |
| **RMSE** | 0.0 | Zero root mean squared error |
| **R² Score** | 1.0 | Perfect fit (100% variance explained) |

> **Predicted salary for 12 years experience = ₹1,39,999** (≈ ₹1,40,000)

The perfect scores are expected because the data has a **perfect linear relationship** — this was an educational example.

---

## 💻 Implementation 2 — Student Performance Prediction (Real Dataset)

### Dataset Overview

**File:** `StudentPerformanceFactors.csv`

| Property | Value |
|----------|-------|
| Rows | 6,607 |
| Columns | 20 |
| Target Variable | `Exam_Score` |
| Task | Predict student exam scores |

### Features

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `Hours_Studied` | Numerical | Daily study hours |
| 2 | `Attendance` | Numerical | Attendance percentage |
| 3 | `Parental_Involvement` | Categorical | Low / Medium / High |
| 4 | `Access_to_Resources` | Categorical | Low / Medium / High |
| 5 | `Extracurricular_Activities` | Categorical | Yes / No |
| 6 | `Sleep_Hours` | Numerical | Average sleep hours |
| 7 | `Previous_Scores` | Numerical | Score in previous exam |
| 8 | `Motivation_Level` | Categorical | Low / Medium / High |
| 9 | `Internet_Access` | Categorical | Yes / No |
| 10 | `Tutoring_Sessions` | Numerical | Number of tutoring sessions |
| 11 | `Family_Income` | Categorical | Low / Medium / High |
| 12 | `Teacher_Quality` | Categorical | Low / Medium / High |
| 13 | `School_Type` | Categorical | Public / Private |
| 14 | `Peer_Influence` | Categorical | Positive / Negative / Neutral |
| 15 | `Physical_Activity` | Numerical | Hours of physical activity/week |
| 16 | `Learning_Disabilities` | Categorical | Yes / No |
| 17 | `Parental_Education_Level` | Categorical | High School / College / Postgraduate |
| 18 | `Distance_from_Home` | Categorical | Near / Moderate / Far |
| 19 | `Gender` | Categorical | Male / Female |
| 20 | `Exam_Score` | **Target** | Final exam score (55–101) |

### Statistical Summary

```
              Hours_Studied  Attendance  Sleep_Hours  Previous_Scores  Exam_Score
count           6607.000000  6607.00000  6607.000000      6607.000000  6607.000000
mean              19.975329    79.977448     7.029060        75.070531    67.235659
std                5.990594    11.547475     1.468120        14.399784     3.890456
min                1.000000    60.000000     4.000000        50.000000    55.000000
25%               16.000000    70.000000     6.000000        63.000000    65.000000
50%               20.000000    80.000000     7.000000        75.000000    67.000000
75%               24.000000    90.000000     8.000000        88.000000    69.000000
max               44.000000   100.000000    10.000000       100.000000   101.000000
```

### Missing Values Handled

| Column | Missing |
|--------|---------|
| `Teacher_Quality` | 78 |
| `Parental_Education_Level` | 90 |
| `Distance_from_Home` | 67 |

**Strategy:**
- **Categorical columns** → filled with **mode** (most frequent value)
- **Numerical columns** → filled with **median**

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, DataFrames |
| `numpy` | Numerical computations, array operations |
| `matplotlib` | Basic plotting and visualization |
| `seaborn` | Statistical data visualization |
| `sklearn.model_selection` | `train_test_split` |
| `sklearn.preprocessing` | `LabelEncoder`, `StandardScaler` |
| `sklearn.linear_model` | `LinearRegression` |
| `sklearn.metrics` | `mean_absolute_error`, `mean_squared_error`, `r2_score` |

---

## ▶️ How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook
```bash
jupyter notebook linearReg.ipynb
```

Or open it in **VS Code**, **Google Colab**, or **JupyterLab**.

---

## 📌 Summary

| Aspect | Simple Example | Real Dataset (Student) |
|--------|---------------|----------------------|
| **Features** | 1 (YearsExperience) | 19 features |
| **Target** | Salary | Exam_Score |
| **Dataset Size** | 10 rows | 6,607 rows |
| **R² Score** | 1.0 (perfect) | Real-world accuracy |
| **Purpose** | Understand the algorithm | Apply to real data |

> 💡 **Linear Regression** works best when:
> - The relationship between features and target is **linear**
> - There's **no multicollinearity** (features aren't highly correlated with each other)
> - Residuals are **normally distributed**
> - **Homoscedasticity** — variance of residuals is constant

---

## 🔗 References

- [Scikit-learn LinearRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Understanding R² Score](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---
👨‍💻 Author

Muhammad Zohaib

⭐ If you found this project useful, consider giving the repository a star.

<p align="center"> <b>Learn → Build → Experiment → Improve 🚀</b> </p>
