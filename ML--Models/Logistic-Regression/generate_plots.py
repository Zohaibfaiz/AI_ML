import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import os

# Create images folder
os.makedirs("images", exist_ok=True)

# ── Load & Preprocess ──────────────────────────────────────
df = pd.read_csv("Social_Network_Ads.csv")
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df.drop(columns=["User ID"], inplace=True)

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_sc, y_train)

y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]

# ── Styling ────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"blue": "#4C72B0", "orange": "#DD8452", "green": "#55A868", "red": "#C44E52"}

# ── 1. Target Class Distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts()
bars = ax.bar(["Not Purchased (0)", "Purchased (1)"], counts.values,
              color=[COLORS["blue"], COLORS["orange"]], edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Count")
ax.set_ylim(0, 300)
plt.tight_layout()
plt.savefig("images/class_distribution.png", dpi=150)
plt.close()
print("✅ class_distribution.png")

# ── 2. Age Distribution by Purchase ───────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
df_orig = pd.read_csv("Social_Network_Ads.csv")
for label, color, name in [(0, COLORS["blue"], "Not Purchased"), (1, COLORS["orange"], "Purchased")]:
    ax.hist(df_orig[df_orig["Purchased"] == label]["Age"],
            bins=20, alpha=0.7, color=color, label=name, edgecolor="white")
ax.set_title("Age Distribution by Purchase Decision", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("images/age_distribution.png", dpi=150)
plt.close()
print("✅ age_distribution.png")

# ── 3. Salary Distribution by Purchase ────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for label, color, name in [(0, COLORS["blue"], "Not Purchased"), (1, COLORS["orange"], "Purchased")]:
    ax.hist(df_orig[df_orig["Purchased"] == label]["EstimatedSalary"],
            bins=20, alpha=0.7, color=color, label=name, edgecolor="white")
ax.set_title("Salary Distribution by Purchase Decision", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Estimated Salary")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("images/salary_distribution.png", dpi=150)
plt.close()
print("✅ salary_distribution.png")

# ── 4. Confusion Matrix ────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Not Purchased", "Purchased"],
            yticklabels=["Not Purchased", "Purchased"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Actual Label", fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150)
plt.close()
print("✅ confusion_matrix.png")

# ── 5. ROC Curve ──────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=COLORS["blue"], lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["blue"])
ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/roc_curve.png", dpi=150)
plt.close()
print("✅ roc_curve.png")

# ── 6. Feature Coefficients ────────────────────────────────
features = ["Gender", "Age", "EstimatedSalary"]
coeffs   = model.coef_[0]
sorted_idx = np.argsort(coeffs)
fig, ax = plt.subplots(figsize=(6, 4))
colors = [COLORS["green"] if c > 0 else COLORS["red"] for c in coeffs[sorted_idx]]
bars = ax.barh([features[i] for i in sorted_idx], coeffs[sorted_idx],
               color=colors, edgecolor="white", height=0.5)
for bar, val in zip(bars, coeffs[sorted_idx]):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=11)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Feature Coefficients", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("images/feature_coefficients.png", dpi=150)
plt.close()
print("✅ feature_coefficients.png")

# ── 7. Metrics Bar Chart ───────────────────────────────────
metrics = {
    "Accuracy" : accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall"   : recall_score(y_test, y_pred),
    "F1 Score" : f1_score(y_test, y_pred),
    "ROC-AUC"  : roc_auc_score(y_test, y_proba),
}
fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"], "#9B59B6"]
bars = ax.bar(metrics.keys(), metrics.values(), color=bar_colors, edgecolor="white", width=0.55)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_title("Model Evaluation Metrics", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("images/evaluation_metrics.png", dpi=150)
plt.close()
print("✅ evaluation_metrics.png")

# ── 8. Age vs Salary Scatter (Decision Boundary proxy) ────
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(df_orig["Age"], df_orig["EstimatedSalary"],
                     c=df_orig["Purchased"], cmap="coolwarm",
                     alpha=0.7, edgecolors="white", s=60)
legend_handles = [
    mpatches.Patch(color="#4393C3", label="Not Purchased"),
    mpatches.Patch(color="#D6604D", label="Purchased"),
]
ax.legend(handles=legend_handles)
ax.set_title("Age vs Estimated Salary (by Purchase)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Age")
ax.set_ylabel("Estimated Salary")
plt.tight_layout()
plt.savefig("images/age_vs_salary.png", dpi=150)
plt.close()
print("✅ age_vs_salary.png")

print("\n🎉 All plots saved to images/ folder!")
