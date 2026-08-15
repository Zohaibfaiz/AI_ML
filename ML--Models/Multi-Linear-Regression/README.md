# MULTI-LINEAR-REG — Employee Salary Prediction

> **AI / ML Project | Domain: Human Resource**

---

## 📌 Project Overview

Is project mein hum employee salaries predict karte hain multiple regression algorithms ka use karke. Dataset mein employees ki salary, gender, age, aur PhD degree ka data hai. Dataset **UCI Machine Learning Repository** se liya gaya hai aur **Kaggle** par bhi available hai.

---

## 📂 Folder Structure

```
Multi-Linear-Regression/
│
└── employee-salaray-prediction-best-ml-algorithms.ipynb   ← Main Jupyter Notebook
```

---

## 📊 Dataset Information

| Property              | Detail             |
|-----------------------|--------------------|
| **Instances**         | 100 samples        |
| **Features**          | 4 (including target) |
| **Target Variable**   | Salary             |
| **Source**            | UCI ML Repository / Kaggle |

### Features (Columns)

| Column   | Type        | Description                         |
|----------|-------------|-------------------------------------|
| `Salary` | float64     | Employee salary (target variable)   |
| `Gender` | int64 (0/1) | Gender — 0: Female, 1: Male         |
| `Age`    | int64       | Age of the employee (20–77)         |
| `PhD`    | int64 (0/1) | PhD degree — 0: No, 1: Yes          |

### Dataset Stats

| Stat   | Salary     | Gender | Age    | PhD  |
|--------|------------|--------|--------|------|
| Mean   | 52.52      | 0.50   | 46.88  | 0.39 |
| Std    | 42.22      | 0.50   | 15.27  | 0.49 |
| Min    | 0.25       | 0.00   | 20.00  | 0.00 |
| Max    | 190.00     | 1.00   | 77.00  | 1.00 |

---

## 🎯 Objective

- Dataset ko explore aur cleanup karna (agar zaroorat ho)
- Multiple regression models build karna employee salary predict karne ke liye
- Models ko evaluate karna aur unke scores (R², RMSE, MAE, etc.) compare karna

---

## 🗂️ Strategic Plan of Action

1. **Data Exploration** — Dataset ko load karna aur basic info dekhna
2. **Exploratory Data Analysis (EDA)** — Visual analysis aur patterns dhundhna
3. **Data Pre-processing** — Missing values, encoding, scaling
4. **Data Manipulation** — Feature engineering
5. **Feature Selection / Extraction** — RFE, VIF analysis
6. **Predictive Modelling** — Multiple ML models train karna
7. **Project Outcomes & Conclusion** — Results aur comparison

---

## 🤖 ML Models Used

| Model                      | Library      |
|----------------------------|--------------|
| Linear Regression          | scikit-learn |
| Ridge Regression           | scikit-learn |
| Lasso Regression           | scikit-learn |
| ElasticNet Regression      | scikit-learn |
| Polynomial Regression      | scikit-learn |
| PCA + Regression           | scikit-learn |

---

## 📦 Libraries / Dependencies

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
IPython
```

Install karne ke liye:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

---

## 🚀 How to Run

1. Repository clone karo:
   ```bash
   git clone https://github.com/<your-username>/AI_ML.git
   ```

2. Notebook open karo:
   ```bash
   cd ML--Models/Multi-Linear-Regression
   jupyter notebook employee-salaray-prediction-best-ml-algorithms.ipynb
   ```

3. Cells ko sequentially run karo.

---

## 📈 Evaluation Metrics

- **R² Score** — Model ka goodness-of-fit
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error

---

## 🙏 Acknowledgements

- Dataset: [Kaggle — Employee Salaries Dataset](https://www.kaggle.com/)
- Original Source: UCI Machine Learning Repository

---

## 📝 Notes

Dataset mein sirf **100 samples** aur **3 features** hain, jo is problem ko challenging banata hai — limited data mein generalizable model banana mushkil hota hai. Is notebook mein explore kiya gaya hai ke kaun sa model best generalization deta hai.

---

*Made with ❤️ for GitHub Upload*
