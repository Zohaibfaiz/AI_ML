import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

print("Starting artifact generation...")

# Set style for high-quality plots
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

df = pd.read_csv('Heart-dis.csv')
print('Original shape:', df.shape)
print('Duplicates:', df.duplicated().sum())

df_clean = df.drop_duplicates().reset_index(drop=True)
print('Cleaned shape:', df_clean.shape)

# Target Distribution Plot
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='target', data=df_clean, palette=['#3498db', '#e74c3c'])
plt.title('Heart Disease Target Distribution (0: Healthy, 1: Disease)', pad=15, fontweight='bold')
plt.xlabel('Diagnosis Target')
plt.ylabel('Patient Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300)
plt.close()
print("Saved target_distribution.png")

# Correlation Heatmap Plot
plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(df_clean.corr(), dtype=bool))
sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, mask=mask)
plt.title('Feature Correlation Heatmap', pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()
print("Saved correlation_heatmap.png")

# Train/Test Split
X = df_clean.drop('target', axis=1)
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

rf_model.fit(X_train, y_train)

train_acc = rf_model.score(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)
cv_scores = cross_val_score(rf_model, X, y, cv=5)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f'Train Acc: {train_acc:.4f}')
print(f'Test Acc: {test_acc:.4f}')
print(f'CV Scores: {cv_scores}')
print(f'Mean CV Acc: {cv_scores.mean():.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 16, 'weight': 'bold'},
            xticklabels=['No Disease (0)', 'Disease (1)'],
            yticklabels=['No Disease (0)', 'Disease (1)'])
plt.xlabel('Predicted Label', labelpad=10, fontweight='bold')
plt.ylabel('Actual Label', labelpad=10, fontweight='bold')
plt.title('Confusion Matrix', pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("Saved confusion_matrix.png")

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2ecc71', lw=3, label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random Chance (AUC = 0.500)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', pad=15, fontweight='bold')
plt.legend(loc='lower right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()
print("Saved roc_curve.png")

# Feature Importance Plot
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 7))
ax = sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Random Forest Feature Importance', pad=15, fontweight='bold')
plt.xlabel('Gini Importance Score')
plt.ylabel('Features')
for p in ax.patches:
    ax.annotate(f'{p.get_width():.3f}', (p.get_width() + 0.002, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("Saved feature_importance.png")

# Save trained model
joblib.dump(rf_model, 'heart_disease_random_forest.pkl')
print('Model dumped successfully!')
