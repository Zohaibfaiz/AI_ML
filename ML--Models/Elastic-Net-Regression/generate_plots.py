import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('housing.csv')
print("Data loaded:", df.shape)

# Fix missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Encode categorical
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False)

# Ensure no NaNs remain
df = df.fillna(df.median(numeric_only=True))

# ─────────────────────────────────────────────
# 1. Correlation Heatmap
# ─────────────────────────────────────────────
numeric_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income',
                'median_house_value']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            annot=True, fmt=".2f", square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap – California Housing Features", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# ─────────────────────────────────────────────
# 2. Distribution of Target Variable
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df['median_house_value'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title("Distribution of Median House Value", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Median House Value ($)", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)

axes[1].hist(np.log1p(df['median_house_value']), bins=50, color='coral', edgecolor='white', alpha=0.85)
axes[1].set_title("Log-Transformed Median House Value", fontsize=13, fontweight='bold')
axes[1].set_xlabel("log(Median House Value + 1)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)

plt.suptitle("Target Variable Distribution", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: target_distribution.png")

# ─────────────────────────────────────────────
# 3. Median Income vs House Value Scatter
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df['median_income'], df['median_house_value'],
                     alpha=0.15, s=8, c=df['median_house_value'],
                     cmap='viridis')
plt.colorbar(scatter, ax=ax, label='Median House Value ($)')
ax.set_xlabel("Median Income (in $10,000s)", fontsize=12)
ax.set_ylabel("Median House Value ($)", fontsize=12)
ax.set_title("Median Income vs. Median House Value", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/income_vs_house_value.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: income_vs_house_value.png")

# ─────────────────────────────────────────────
# 4. Geographic Heatmap (Lat/Lon)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df['longitude'], df['latitude'],
                     c=df['median_house_value'], cmap='hot_r',
                     alpha=0.4, s=5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Median House Value ($)', fontsize=11)
ax.set_title("California Housing Prices – Geographic Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
plt.tight_layout()
plt.savefig('images/geographic_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: geographic_heatmap.png")

# ─────────────────────────────────────────────
# 5. Elastic Net – Actual vs Predicted
# ─────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != 'median_house_value']
X = df[feature_cols]
y = df['median_house_value']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred, alpha=0.3, s=12, color='royalblue', label='Predictions')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel("Actual Median House Value ($)", fontsize=12)
ax.set_ylabel("Predicted Median House Value ($)", fontsize=12)
ax.set_title(f"Elastic Net – Actual vs. Predicted\nR² = {r2:.4f}  |  RMSE = ${rmse:,.0f}", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('images/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: actual_vs_predicted.png  (R²={r2:.4f}, RMSE={rmse:.0f})")

# ─────────────────────────────────────────────
# 6. Residual Plot
# ─────────────────────────────────────────────
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(y_pred, residuals, alpha=0.3, s=10, color='darkorange')
ax.axhline(0, color='red', linestyle='--', lw=2)
ax.set_xlabel("Predicted Median House Value ($)", fontsize=12)
ax.set_ylabel("Residuals ($)", fontsize=12)
ax.set_title("Elastic Net – Residual Plot", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/residual_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: residual_plot.png")

# ─────────────────────────────────────────────
# 7. Elastic Net – L1 Ratio Effect on Coefficients
# ─────────────────────────────────────────────
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
coef_matrix = []
for r in l1_ratios:
    m = ElasticNet(alpha=0.1, l1_ratio=r, max_iter=10000, random_state=42)
    m.fit(X_train, y_train)
    coef_matrix.append(m.coef_)

coef_df = pd.DataFrame(coef_matrix, columns=feature_cols, index=[str(r) for r in l1_ratios])

fig, ax = plt.subplots(figsize=(13, 6))
coef_df.T.plot(ax=ax, marker='o', linewidth=1.5)
ax.set_title("Effect of L1 Ratio on Elastic Net Coefficients", fontsize=14, fontweight='bold')
ax.set_xlabel("Feature", fontsize=11)
ax.set_ylabel("Coefficient Value", fontsize=11)
ax.legend(title="l1_ratio", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('images/l1_ratio_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: l1_ratio_effect.png")

# ─────────────────────────────────────────────
# 8. Ocean Proximity – Average House Value
# ─────────────────────────────────────────────
df_orig = pd.read_csv('housing.csv')
df_orig["total_bedrooms"].fillna(df_orig["total_bedrooms"].median(), inplace=True)
avg_price = df_orig.groupby('ocean_proximity')['median_house_value'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = avg_price.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(avg_price)), edgecolor='white', width=0.65)
ax.set_title("Average Median House Value by Ocean Proximity", fontsize=14, fontweight='bold')
ax.set_xlabel("Ocean Proximity", fontsize=12)
ax.set_ylabel("Average Median House Value ($)", fontsize=12)
ax.tick_params(axis='x', rotation=30)
for p in ax.patches:
    ax.annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('images/ocean_proximity_avg_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ocean_proximity_avg_price.png")

print("\nAll plots generated successfully!")
print(f"Final model metrics -> R²: {r2:.4f}, RMSE: ${rmse:,.0f}")
