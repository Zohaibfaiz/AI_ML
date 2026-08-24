import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Create images folder
os.makedirs('images', exist_ok=True)

print("--- Generating Graphs for KNN Model ---")

# ==========================================
# 1. DIABETES DATASET ANALYSIS & GRAPHS
# ==========================================
df_diab = pd.read_csv('diabetes.csv')

# Handle missing zero values in specific columns
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df_diab[col] = df_diab[col].replace(0, np.nan)
    df_diab[col] = df_diab[col].fillna(df_diab[col].median())

# 1.1 Class Distribution Chart
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2ecc71', '#e74c3c']

# Bar chart
sns.countplot(x='Outcome', data=df_diab, palette=colors, ax=ax[0])
ax[0].set_title('Diabetes Target Class Count', fontsize=14, fontweight='bold', pad=15)
ax[0].set_xticklabels(['Not Diabetic (0)', 'Diabetic (1)'])
ax[0].set_xlabel('Class Outcome', fontsize=11)
ax[0].set_ylabel('Count', fontsize=11)

for p in ax[0].patches:
    ax[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=11, fontweight='bold')

# Pie chart
counts = df_diab['Outcome'].value_counts()
ax[1].pie(counts, labels=['Not Diabetic (0)', 'Diabetic (1)'], autopct='%1.1f%%',
          startangle=90, colors=colors, explode=(0.05, 0), shadow=True,
          textprops={'fontsize': 11, 'fontweight': 'bold'})
ax[1].set_title('Class Percentage Split', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('images/diabetes_class_dist.png', dpi=300)
plt.close()
print("Saved: images/diabetes_class_dist.png")

# 1.2 Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df_diab.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Diabetes Dataset - Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('images/diabetes_corr_heatmap.png', dpi=300)
plt.close()
print("Saved: images/diabetes_corr_heatmap.png")

# Preprocessing & Model Training for Diabetes
X = df_diab.drop('Outcome', axis=1)
y = df_diab['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

K = 5
knn_diab = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
knn_diab.fit(X_train_scaled, y_train)
y_pred = knn_diab.predict(X_test_scaled)

# 1.3 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            annot_kws={"size": 16, "weight": "bold"},
            xticklabels=['Predicted:\nNot Diabetic', 'Predicted:\nDiabetic'],
            yticklabels=['Actual:\nNot Diabetic', 'Actual:\nDiabetic'])
plt.title(f'Confusion Matrix - KNN (K={K})', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('images/diabetes_confusion_matrix.png', dpi=300)
plt.close()
print("Saved: images/diabetes_confusion_matrix.png")

# 1.4 K Value Tuning (K vs Accuracy)
k_range = list(range(1, 26, 2))
train_acc = []
test_acc = []

for k in k_range:
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    clf.fit(X_train_scaled, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train_scaled)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test_scaled)))

best_k = k_range[np.argmax(test_acc)]
best_acc = max(test_acc)

plt.figure(figsize=(10, 5))
plt.plot(k_range, train_acc, 'o--', color='#3498db', label='Training Accuracy', linewidth=2)
plt.plot(k_range, test_acc, 'o-', color='#e67e22', label='Testing Accuracy', linewidth=2.5)
plt.axvline(x=best_k, color='#e74c3c', linestyle=':', label=f'Optimal K = {best_k} ({best_acc*100:.1f}%)')
plt.title('Diabetes Dataset - K Value vs Model Accuracy', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xticks(k_range)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('images/diabetes_k_vs_accuracy.png', dpi=300)
plt.close()
print("Saved: images/diabetes_k_vs_accuracy.png")

# 1.5 Decision Boundary (2D: Glucose vs BMI)
X_2d = df_diab[['Glucose', 'BMI']].values
y_2d = df_diab['Outcome'].values

X_tr_2d, X_te_2d, y_tr_2d, y_te_2d = train_test_split(X_2d, y_2d, test_size=0.2, random_state=42, stratify=y_2d)

scaler_2d = StandardScaler()
X_tr_2d_s = scaler_2d.fit_transform(X_tr_2d)
X_te_2d_s = scaler_2d.transform(X_te_2d)

model_2d = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model_2d.fit(X_tr_2d_s, y_tr_2d)

h = 0.05
x_min, x_max = X_tr_2d_s[:, 0].min() - 1, X_tr_2d_s[:, 0].max() + 1
y_min, y_max = X_tr_2d_s[:, 1].min() - 1, X_tr_2d_s[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
plt.contour(xx, yy, Z, colors='k', linewidths=0.5)

for cls, label, color in [(0, 'Not Diabetic', '#c0392b'), (1, 'Diabetic', '#27ae60')]:
    idx = (y_tr_2d == cls)
    plt.scatter(X_tr_2d_s[idx, 0], X_tr_2d_s[idx, 1], c=color, label=label, edgecolors='black', s=45, alpha=0.75)

plt.title('KNN Decision Boundary (K=5)\nFeatures: Glucose vs BMI (Scaled Space)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Glucose (Standardized)', fontsize=12)
plt.ylabel('BMI (Standardized)', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('images/diabetes_decision_boundary.png', dpi=300)
plt.close()
print("Saved: images/diabetes_decision_boundary.png")


# ==========================================
# 2. GLASS DATASET ANALYSIS & GRAPHS
# ==========================================
df_glass = pd.read_csv('glass.csv')
df_glass.drop_duplicates(inplace=True)

X_g = df_glass.drop('Type', axis=1)
y_g = df_glass['Type']

X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(X_g, y_g, test_size=0.2, random_state=20)

m_scaler = MinMaxScaler()
X_g_train_s = m_scaler.fit_transform(X_g_train)
X_g_test_s = m_scaler.transform(X_g_test)

glass_k_range = list(range(1, 16))
glass_acc = []

for k in glass_k_range:
    knn_g = KNeighborsClassifier(n_neighbors=k)
    knn_g.fit(X_g_train_s, y_g_train)
    pred_g = knn_g.predict(X_g_test_s)
    glass_acc.append(accuracy_score(y_g_test, pred_g))

best_g_k = glass_k_range[np.argmax(glass_acc)]
best_g_acc = max(glass_acc)

plt.figure(figsize=(9, 5))
plt.plot(glass_k_range, glass_acc, marker='o', color='#8e44ad', markerfacecolor='#2c3e50', markeredgecolor='#27ae60', markersize=8, linewidth=2)
plt.axvline(x=best_g_k, color='#e74c3c', linestyle='--', label=f'Optimal K = {best_g_k} (Accuracy: {best_g_acc*100:.2f}%)')
plt.title('Glass Identification Dataset - K vs Accuracy', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(glass_k_range)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('images/glass_k_vs_accuracy.png', dpi=300)
plt.close()
print("Saved: images/glass_k_vs_accuracy.png")

print("--- All Graphs Successfully Generated! ---")
