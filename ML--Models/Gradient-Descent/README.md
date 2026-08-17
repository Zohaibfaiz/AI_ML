# 📉 Gradient Descent — Linear Regression from Scratch

A clean, step-by-step implementation of **Gradient Descent** for **Linear Regression** built entirely from scratch using Python — no scikit-learn, no shortcuts.

---

## 🚀 Overview

This project demonstrates how the **Gradient Descent** optimization algorithm works under the hood to train a Linear Regression model. Instead of using high-level ML libraries, every step is implemented manually so you can clearly understand what's happening at each stage.

The model learns the relationship `y = 2x` from a simple dataset and converges to near-perfect predictions after 1000 epochs.

---

## 📊 Dataset

A simple synthetic dataset is used:

| X | y (Actual) |
|---|-----------|
| 1 | 2         |
| 2 | 4         |
| 3 | 6         |
| 4 | 8         |
| 5 | 10        |

The true underlying pattern is `y = 2x` (weight = 2, bias = 0).

---

## 🧠 How It Works

The notebook follows a clear 13-step pipeline:

### Step 1 — Dataset
Define the input features `X` and target labels `y` as NumPy arrays.

### Step 2 — Initialize Parameters
Set initial weight `w = 0.0` and bias `b = 0.0`.

```
learning_rate = 0.01
epochs        = 1000
```

### Step 3 — Gradient Descent Loop
For each epoch:

1. **Prediction** → `y_pred = w * X + b`
2. **Error** → `error = y_pred - y`
3. **Cost (MSE)** → `cost = mean(error²)`
4. **Gradients** →
   - `dw = (2/n) * Σ(error * X)`
   - `db = (2/n) * Σ(error)`
5. **Update Parameters** →
   - `w = w - lr * dw`
   - `b = b - lr * db`

### Step 4 — Final Results
Print the learned weight, bias, and the final regression equation.

### Step 5 — Prediction Table
Display actual vs predicted values with error column using Pandas.

### Step 6 — Final MSE
Compute the final Mean Squared Error after training.

### Step 7 — Convergence Check
Automatically check if the model converged (last cost < first cost).

### Step 8–12 — Visualizations
- 📈 Cost vs Epoch curve
- 📉 Regression Line over data points
- 🎯 Actual vs Predicted scatter plot
- 🔍 Residual Plot
- 📊 Residual Distribution Histogram

### Step 13 — Cost Reduction Summary
Print first and last cost values to show how much the model improved.

---

## 📈 Training Output

```
==================================================
Gradient Descent Training Started
==================================================
Epoch    0 | Cost = 44.000000
Epoch  100 | Cost = 0.024474
Epoch  200 | Cost = 0.012432
Epoch  300 | Cost = 0.006315
Epoch  400 | Cost = 0.003208
Epoch  500 | Cost = 0.001630
Epoch  600 | Cost = 0.000828
Epoch  700 | Cost = 0.000420
Epoch  800 | Cost = 0.000214
Epoch  900 | Cost = 0.000108

==================================================
Training Finished
==================================================

Final Weight (w) : 1.9952
Final Bias   (b) : 0.0174

Final Linear Regression Equation:
y = 1.9952x + 0.0174

Prediction Table:
     X  Actual  Predicted  Error
0  1.0     2.0       2.01  -0.01
1  2.0     4.0       4.01  -0.01
2  3.0     6.0       6.00  -0.00
3  4.0     8.0       8.00   0.00
4  5.0    10.0       9.99   0.01

Final MSE : 5.5e-05

Convergence Check:
Model Converged Successfully
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| NumPy | Array operations & math |
| Matplotlib | Visualizations |
| Pandas | Prediction table display |
| Google Colab | Development environment |

---

## 📂 File Structure

```
Gradient-Descent/
│
└── Gradient.ipynb    # Main notebook with full implementation
```

---

## ▶️ How to Run

**Option 1 — Google Colab (Recommended)**
1. Upload `Gradient.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Click `Runtime` → `Run All`

**Option 2 — Locally**
```bash
# Install dependencies
pip install numpy matplotlib pandas jupyter

# Run notebook
jupyter notebook Gradient.ipynb
```

---

## 📌 Key Concepts Covered

- ✅ What is Gradient Descent?
- ✅ Cost Function (Mean Squared Error)
- ✅ Partial Derivatives (Gradients)
- ✅ Parameter Update Rule
- ✅ Convergence and Training Progress
- ✅ Model Evaluation (MSE, Residuals)
- ✅ Visualization of training process

---

## 🎯 Results

The model successfully learns the linear relationship `y ≈ 2x`:
- **Final Weight:** `1.9952` (close to true value `2.0`)
- **Final Bias:** `0.0174` (close to true value `0.0`)
- **Final MSE:** `0.000055` (near-zero error)

---

## 👤 Author

Feel free to explore, fork, and learn from this project!

⭐ If this helped you understand Gradient Descent, consider giving it a star!
