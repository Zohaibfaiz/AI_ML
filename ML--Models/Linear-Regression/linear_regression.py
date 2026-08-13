"""
============================================================
  LINEAR REGRESSION — HOUSE PRICE PREDICTION
  Author  : Zohaib | FA23-BSE-048
  Dataset : dataset.csv (7000 rows, 20 features)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  LINEAR REGRESSION — HOUSE PRICE PREDICTION")
print("=" * 55)

df = pd.read_csv('dataset.csv')
print(f"\n[✔] Dataset loaded : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print(f"    Columns         : {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("[→] Basic Statistics:")
print(df[['price', 'sqft_living', 'bedrooms', 'bathrooms', 'grade']].describe().round(2))
print(f"\n[→] Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ─────────────────────────────────────────────
# 3.  DATA CLEANING
# ─────────────────────────────────────────────
print("\n[✔] Cleaning data...")

# Drop id — not a feature
df.drop('id', axis=1, inplace=True)

# Fill missing numeric values with median
for col in df.select_dtypes(include=[np.number]).columns:
    median = df[col].median()
    missing = df[col].isnull().sum()
    if missing > 0:
        df[col].fillna(median, inplace=True)
        print(f"    Filled {missing:>3} NaN in '{col}' with median={median}")

# Remove extreme outliers in price using IQR
Q1, Q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
IQR     = Q3 - Q1
before  = len(df)
df      = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
print(f"    Outliers removed : {before - len(df)} rows  |  Remaining: {len(df):,} rows")

# ─────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
df['house_age']       = 2024 - df['yr_built']
df['was_renovated']   = (df['yr_renovated'] > 0).astype(int)
df['total_sqft']      = df['sqft_living'] + df['sqft_basement']
df['price_per_sqft']  = df['price'] / df['sqft_living']
print("\n[✔] New features added: house_age, was_renovated, total_sqft, price_per_sqft")

# ─────────────────────────────────────────────
# 5.  FEATURES & TARGET
# ─────────────────────────────────────────────
FEATURES = [
    'sqft_living', 'bedrooms', 'bathrooms', 'floors',
    'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'sqft_living15',
    'house_age', 'was_renovated', 'total_sqft'
]
TARGET = 'price'

X = df[FEATURES].copy()
y = df[TARGET]

# Final safety: fill any remaining NaN in features
X = X.fillna(X.median())
print(f"\n[✔] Features ({len(FEATURES)}): {FEATURES}")
print(f"    Target          : {TARGET}")
print(f"    Any NaN left?   : {X.isnull().any().any()}")

# ─────────────────────────────────────────────
# 6.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n[✔] Train : {X_train.shape[0]:,} samples")
print(f"    Test  : {X_test.shape[0]:,} samples  (80 / 20 split)")

# ─────────────────────────────────────────────
# 7.  SCALING
# ─────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n[✔] Features scaled with StandardScaler")

# ─────────────────────────────────────────────
# 8.  TRAIN MODEL
# ─────────────────────────────────────────────
print("\n[→] Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("[✔] Training complete!")

# ─────────────────────────────────────────────
# 9.  EVALUATION
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
r2     = r2_score(y_test, y_pred)
mae    = mean_absolute_error(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)
rmse   = np.sqrt(mse)

print("\n" + "=" * 45)
print("  MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"  R² Score   :  {r2:.4f}  ({r2*100:.2f}% variance explained)")
print(f"  MAE        :  ${mae:>12,.0f}")
print(f"  MSE        :  ${mse:>12,.0f}")
print(f"  RMSE       :  ${rmse:>12,.0f}")
print(f"  Intercept  :  {model.intercept_:,.2f}")
print("=" * 45)

# Feature coefficients
coef_df = pd.DataFrame({
    'Feature'     : FEATURES,
    'Coefficient' : model.coef_
}).sort_values('Coefficient', ascending=False)
print("\n[✔] Feature Coefficients (sorted):")
print(coef_df.to_string(index=False))

# ─────────────────────────────────────────────
# 10. SAVE MODEL
# ─────────────────────────────────────────────
bundle = {'model': model, 'scaler': scaler, 'features': FEATURES}
with open('model.pkl', 'wb') as f:
    pickle.dump(bundle, f)
print("\n[✔] model.pkl saved successfully!")

# ─────────────────────────────────────────────
# 11. PREDICTION VISUALIZATION
# ─────────────────────────────────────────────
print("\n[→] Generating prediction.png ...")

sns.set_theme(style='darkgrid', palette='muted')
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

TITLE_COLOR = '#e6edf3'
ACCENT1     = '#58a6ff'
ACCENT2     = '#f78166'
ACCENT3     = '#56d364'
ACCENT4     = '#d2a8ff'

def style_ax(ax, title):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color=TITLE_COLOR, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')

# ── Plot 1: Actual vs Predicted ──────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test / 1e6, y_pred / 1e6, alpha=0.35, color=ACCENT1, s=10, edgecolors='none')
lims = [min(y_test.min(), y_pred.min()) / 1e6, max(y_test.max(), y_pred.max()) / 1e6]
ax1.plot(lims, lims, '--', color=ACCENT2, lw=1.5, label='Perfect fit')
ax1.set_xlabel('Actual Price (M$)')
ax1.set_ylabel('Predicted Price (M$)')
ax1.legend(fontsize=8, labelcolor='white', facecolor='#21262d')
style_ax(ax1, f'Actual vs Predicted  (R²={r2:.3f})')

# ── Plot 2: Residuals ────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test - y_pred
ax2.scatter(y_pred / 1e6, residuals / 1e3, alpha=0.35, color=ACCENT3, s=10, edgecolors='none')
ax2.axhline(0, color=ACCENT2, linestyle='--', lw=1.5)
ax2.set_xlabel('Predicted Price (M$)')
ax2.set_ylabel('Residuals (K$)')
style_ax(ax2, 'Residual Plot')

# ── Plot 3: Residual Distribution ────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals / 1e3, bins=50, color=ACCENT4, edgecolor='none', alpha=0.8)
ax3.axvline(0, color=ACCENT2, linestyle='--', lw=1.5)
ax3.set_xlabel('Residuals (K$)')
ax3.set_ylabel('Frequency')
style_ax(ax3, 'Residual Distribution')

# ── Plot 4: Feature Coefficients ─────────────
ax4 = fig.add_subplot(gs[1, 0:2])
colors_bar = [ACCENT3 if v >= 0 else ACCENT2 for v in coef_df['Coefficient']]
bars = ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_bar, edgecolor='none')
ax4.axvline(0, color='white', lw=0.8, alpha=0.4)
ax4.set_xlabel('Coefficient Value')
style_ax(ax4, 'Feature Importance (Coefficients)')

# ── Plot 5: Price Distribution ───────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(y_test / 1e6, bins=40, alpha=0.6, color=ACCENT1, label='Actual', edgecolor='none')
ax5.hist(y_pred / 1e6, bins=40, alpha=0.6, color=ACCENT2, label='Predicted', edgecolor='none')
ax5.set_xlabel('Price (M$)')
ax5.set_ylabel('Frequency')
ax5.legend(fontsize=8, labelcolor='white', facecolor='#21262d')
style_ax(ax5, 'Price Distribution: Actual vs Predicted')

# ── Super title ───────────────────────────────
fig.text(0.5, 0.97, '🏠  House Price Prediction — Linear Regression',
         ha='center', va='top', fontsize=15, fontweight='bold',
         color=TITLE_COLOR)
fig.text(0.5, 0.93, f'Dataset: 7,000 rows  |  Features: {len(FEATURES)}  |  '
         f'R²: {r2:.4f}  |  RMSE: ${rmse:,.0f}',
         ha='center', va='top', fontsize=9, color='#8b949e')

plt.savefig('prediction.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()    # ← yeh add karo
print("[✔] prediction.png saved!")

# ─────────────────────────────────────────────
# 12. PREDICTION ON NEW SAMPLE
# ─────────────────────────────────────────────
print("\n[→] Sample Prediction:")
sample = pd.DataFrame([{
    'sqft_living'   : 2500,
    'bedrooms'      : 4,
    'bathrooms'     : 2.5,
    'floors'        : 2.0,
    'waterfront'    : 0,
    'view'          : 1,
    'condition'     : 4,
    'grade'         : 8,
    'sqft_above'    : 2000,
    'sqft_basement' : 500,
    'sqft_living15' : 2200,
    'house_age'     : 20,
    'was_renovated' : 1,
    'total_sqft'    : 3000,
}])
sample_scaled   = scaler.transform(sample)
sample_pred     = model.predict(sample_scaled)[0]
print(f"    Input  : 2500 sqft | 4 bed | 2.5 bath | Grade 8 | Age 20")
print(f"    Predicted Price : ${sample_pred:,.0f}")

print("\n✅ All done! Files created: model.pkl | prediction.png")
