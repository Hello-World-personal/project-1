import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================
# 1. Load the column names from pkl
# ============================================
with open("/home/topsoe/vrsh/streamlit-d/example/vessel_orientation_gbm.pkl", "rb") as f:
    saved_columns = pickle.load(f)

print("Saved column names from .pkl:")
print(list(saved_columns))

# ============================================
# 2. Load the dataset
# ============================================
df = pd.read_excel("/home/topsoe/vrsh/streamlit-d/example/vessel_input_cleaned.xlsx")
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ============================================
# 3. Prepare features and target
# ============================================
target_col = "VesselOrientation"

# Try case-insensitive match if exact name not found
if target_col not in df.columns:
    for col in df.columns:
        if col.lower() == target_col.lower():
            target_col = col
            break

print(f"\nTarget column: {target_col}")
print(f"Target values: {df[target_col].unique()}")

X = df.drop(columns=[target_col])
y = df[target_col]

# ============================================
# 4. Encode categorical columns
# ============================================
label_encoders = {}
X_encoded = X.copy()

for col in X_encoded.columns:
    if X_encoded[col].dtype == "object":
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded: {col} ({le.classes_.shape[0]} categories)")

# Encode target if needed
if y.dtype == "object":
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y.astype(str))
    print(f"  Encoded target: {target_col} ({target_le.classes_})")
else:
    y_encoded = y

# Handle missing values
X_encoded = X_encoded.fillna(0)

# ============================================
# 5. Train GBM and get feature importance
# ============================================
print("\nTraining GBM model...")
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_encoded, y_encoded)

score = model.score(X_encoded, y_encoded)
print(f"Training accuracy: {score:.4f}")

# ============================================
# 6. Feature importance
# ============================================
importance = model.feature_importances_

results = []
total = sum(importance)
for name, imp in zip(X_encoded.columns, importance):
    pct = round((imp / total) * 100, 2)
    results.append((name, imp, pct))

results.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE")
print("=" * 65)

cumulative = 0
for name, imp, pct in results:
    cumulative += pct
    bar = "█" * int(pct)
    print(f"  {name:35s}  {pct:6.2f}%  {bar}")

# ============================================
# 7. Classify relevant vs irrelevant
# ============================================
threshold_pct = 1.0  # adjust as needed

relevant = [(n, p) for n, i, p in results if p >= threshold_pct]
irrelevant = [(n, p) for n, i, p in results if p < threshold_pct]

print(f"\n{'=' * 65}")
print(f"RELEVANT columns (>= {threshold_pct}%)")
print(f"{'=' * 65}")
for name, pct in relevant:
    print(f"  ✅ {name:35s} → {pct}%")

print(f"\nIRRELEVANT columns (< {threshold_pct}%)")
print(f"{'=' * 65}")
if irrelevant:
    for name, pct in irrelevant:
        print(f"  ❌ {name:35s} → {pct}%")
else:
    print("  None — all columns are above the threshold")

# ============================================
# 8. Summary
# ============================================
print(f"\n{'=' * 65}")
print("SUMMARY")
print(f"{'=' * 65}")
print(f"Total features:      {len(X_encoded.columns)}")
print(f"Relevant:            {len(relevant)}")
print(f"Irrelevant:          {len(irrelevant)}")
print(f"\nKeep: {[n for n, p in relevant]}")
print(f"Drop: {[n for n, p in irrelevant]}")