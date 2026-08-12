"""
Step 2: Preprocessing.

- One-hot encode categorical columns (protocol_type, service, flag)
- Scale numeric columns (StandardScaler)
- Split into train/test sets, stratified so rare classes (e.g. u2r) appear
  in both train and test in proportion.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/home/claude/nids/nids_data.csv")

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COL = "label"

# One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)

X = df_encoded.drop(columns=[LABEL_COL])
y = df_encoded[LABEL_COL]

numeric_cols = X.columns  # after one-hot, everything left is numeric/binary

# Stratified split keeps class proportions consistent across train/test,
# which matters a lot here because u2r/r2l are so rare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on TRAIN ONLY, then apply to both -> avoids leaking test info into training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Save for the next step
X_train_scaled.to_csv("/home/claude/nids/X_train.csv", index=False)
X_test_scaled.to_csv("/home/claude/nids/X_test.csv", index=False)
y_train.to_csv("/home/claude/nids/y_train.csv", index=False)
y_test.to_csv("/home/claude/nids/y_test.csv", index=False)

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)
print("\nTrain class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())
