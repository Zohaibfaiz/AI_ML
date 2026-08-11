# 📈 Linear Regression — ML Model

> A comprehensive implementation of **Linear Regression** using Python and Scikit-learn, covering both a simple from-scratch example and a real-world dataset project.

---

## 📁 Files in This Folder

| File | Description |
|------|-------------|
| `linearReg.ipynb` | Main Jupyter Notebook with two complete implementations |
| `StudentPerformanceFactors.csv` | Real-world dataset (6607 rows × 20 columns) for multi-feature regression |
| `README.md` | This documentation file |

---

## 🧠 What is Linear Regression?

**Linear Regression** is one of the most fundamental supervised machine learning algorithms. It models the **linear relationship** between an independent variable (or multiple variables) and a continuous dependent variable.

### 📐 The Math Behind It

The linear regression equation:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- **ŷ** = Predicted output (dependent variable)
- **β₀** = Intercept (bias term — value of y when all x = 0)
- **β₁, β₂, ..., βₙ** = Coefficients (slopes for each feature)
- **x₁, x₂, ..., xₙ** = Input features (independent variables)

For **Simple Linear Regression** (one feature):

```
ŷ = β₀ + β₁x
```

Which can be rewritten as:

```
y = mx + b
```

Where **m** is the slope and **b** is the intercept.

---

## 🎯 How Does It Work?

The algorithm finds the **best-fit line** by minimizing the **Sum of Squared Errors (SSE)** — the sum of squared distances between actual and predicted values:

```
SSE = Σ (yᵢ - ŷᵢ)²
```

This is solved using the **Ordinary Least Squares (OLS)** method:

```
β = (XᵀX)⁻¹ Xᵀy
```

---

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

### Steps

```python
# STEP 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 2: Create Dataset
data = {
    "YearsExperience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 40000, 50000, 60000, 70000,
               80000, 90000, 100000, 110000, 120000]
}
df = pd.DataFrame(data)

# STEP 3: Features and Target
X = df[["YearsExperience"]]
y = df["Salary"]

# STEP 4: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 5-6: Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 7: Model Parameters
print("Slope   :", model.coef_[0])      # → ~10,000
print("Intercept:", model.intercept_)   # → ~20,000

# STEP 8: Predict
y_pred = model.predict(X_test)

# STEP 9: Evaluate
print("MAE  :", mean_absolute_error(y_test, y_pred))   # → 0.0
print("MSE  :", mean_squared_error(y_test, y_pred))    # → 0.0
print("RMSE :", mse ** 0.5)                            # → 0.0
print("R²   :", r2_score(y_test, y_pred))              # → 1.0

# STEP 11: Predict New Value
new_employee = pd.DataFrame({"YearsExperience": [12]})
print(model.predict(new_employee))  # → ~140,000
```

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

### Full Pipeline Code

```python
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# 3-5. EDA: Basic Info, Missing Values, Duplicates
print(df.shape)          # (6607, 20)
print(df.isnull().sum()) # Check missing values
print(df.duplicated().sum())  # 0 duplicates

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

df.drop_duplicates(inplace=True)

# 6. Visualize Numerical Columns
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    plt.figure(figsize=(5, 3))
    sns.histplot(df[col], kde=True)   # Distribution
    plt.title(col)
    plt.show()

    plt.figure(figsize=(5, 2))
    sns.boxplot(x=df[col])             # Outlier detection
    plt.title(col)
    plt.show()

# 7. Visualize Categorical Columns
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    plt.figure(figsize=(6, 3))
    sns.countplot(x=df[col])
    plt.xticks(rotation=45)
    plt.title(col)
    plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.show()

# 9. Outlier Removal (IQR Method)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# 10. Label Encoding (Categorical → Numerical)
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# 11. Feature Scaling
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 12. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 13-14. Train & Predict
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 15. Evaluate
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)

# 16. Actual vs Predicted Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# 17. Feature Coefficients (Importance)
importance = pd.DataFrame({
    "Feature": df.drop("Exam_Score", axis=1).columns,
    "Coefficient": model.coef_
})
print(importance.sort_values(by="Coefficient", ascending=False))

# Accuracy as percentage
accuracy = r2 * 100
print(f"Model Accuracy: {accuracy:.2f}%")
```

---

## 📊 Evaluation Metrics Explained

### 1. Mean Absolute Error (MAE)
```
MAE = (1/n) × Σ |yᵢ - ŷᵢ|
```
- **Interpretation:** Average absolute difference between actual and predicted values
- **Units:** Same as target variable
- **Lower is better** ✅

### 2. Mean Squared Error (MSE)
```
MSE = (1/n) × Σ (yᵢ - ŷᵢ)²
```
- **Interpretation:** Average squared difference — penalizes large errors more
- **Units:** Square of target variable units
- **Lower is better** ✅

### 3. Root Mean Squared Error (RMSE)
```
RMSE = √MSE
```
- **Interpretation:** Square root of MSE — back to original units
- **Most interpretable error metric**
- **Lower is better** ✅

### 4. R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```
Where:
- `SS_res` = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- `SS_tot` = Σ(yᵢ - ȳ)² (total sum of squares)

| R² Value | Meaning |
|----------|---------|
| **1.0** | Perfect prediction — model explains all variance |
| **0.7–0.9** | Good model |
| **0.5–0.7** | Moderate model |
| **< 0.5** | Weak model |
| **0.0** | Model no better than predicting mean |
| **< 0** | Worse than predicting mean |

---

## 🔑 Key Concepts

### Preprocessing Steps Applied

| Step | Technique | Why |
|------|-----------|-----|
| **Missing Values** | Mode (categorical), Median (numerical) | Preserve distribution without bias |
| **Duplicates** | `drop_duplicates()` | Prevent model overfitting on repeated data |
| **Outlier Removal** | IQR Method (capping) | Reduce extreme value influence |
| **Encoding** | `LabelEncoder` | Convert categories to numbers for model |
| **Scaling** | `StandardScaler` | Normalize features to same scale |

### Train-Test Split
```
Total Data → 80% Training + 20% Testing
              ↓                  ↓
         Learn patterns      Evaluate generalization
```
- `random_state=42` ensures **reproducibility**

### Feature Scaling (StandardScaler)
```
z = (x - μ) / σ
```
- Transforms each feature to have **mean = 0** and **standard deviation = 1**
- Essential for algorithms sensitive to feature scale

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

*Made with ❤️ | Linear Regression Implementation in Python*
