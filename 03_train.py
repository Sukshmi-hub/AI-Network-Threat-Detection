"""
Step 3 & 4: Handle class imbalance + train models.

NOTE ON SUBSTITUTIONS (be upfront about these if asked):
- No internet access in this sandbox -> can't install TensorFlow. Using
  Scikit-learn's MLPClassifier as the neural network instead. It's a real
  feedforward neural net (same core idea as a basic TensorFlow model), just
  a different library. Swap in TensorFlow's Sequential API later if you
  want the resume claim to be literal, using the same train/test split.
- No imbalanced-learn (SMOTE) available -> using class_weight='balanced'
  instead, which is the simpler, still-legitimate first approach: it tells
  the model to penalize mistakes on rare classes (like u2r) more heavily,
  rather than creating synthetic new rows.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib

X_train = pd.read_csv("/home/claude/nids/X_train.csv")
y_train = pd.read_csv("/home/claude/nids/y_train.csv").squeeze()

# --- Model 1: Random Forest, with class_weight='balanced' for imbalance ---
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",   # <-- this is the class-imbalance fix
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
joblib.dump(rf, "/home/claude/nids/model_rf.joblib")
print("Random Forest trained.")

# --- Model 2: Neural network (MLPClassifier stands in for TensorFlow here) ---
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    random_state=42,
)
mlp.fit(X_train, y_train)
joblib.dump(mlp, "/home/claude/nids/model_mlp.joblib")
print("Neural network (MLP) trained.")

print("\nFeature importances (top 10, Random Forest):")
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importances.head(10))
