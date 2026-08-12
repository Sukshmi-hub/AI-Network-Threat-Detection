"""
Step 5: Evaluation.

Prints accuracy/precision/recall/F1 per model, plus a per-class breakdown
(since overall accuracy hides how badly rare classes like u2r perform).
Saves confusion matrix plots for both models.
"""

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)

X_test = pd.read_csv("/home/claude/nids/X_test.csv")
y_test = pd.read_csv("/home/claude/nids/y_test.csv").squeeze()

rf = joblib.load("/home/claude/nids/model_rf.joblib")
mlp = joblib.load("/home/claude/nids/model_mlp.joblib")

LABELS_ORDER = ["normal", "dos", "probe", "r2l", "u2r"]

def evaluate(model, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"Overall accuracy:  {acc:.3f}")
    print(f"Weighted precision: {prec:.3f}")
    print(f"Weighted recall:    {rec:.3f}")
    print(f"Weighted F1:        {f1:.3f}")
    print("\nPer-class report (this is what actually matters for rare classes):")
    print(classification_report(y_test, preds, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=LABELS_ORDER)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS_ORDER, yticklabels=LABELS_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    fname = f"/home/claude/nids/confusion_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved confusion matrix -> {fname}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

results = {}
results["Random Forest"] = evaluate(rf, "Random Forest")
results["Neural Net (MLP)"] = evaluate(mlp, "Neural Net (MLP)")

print(f"\n{'='*50}\nSUMMARY COMPARISON\n{'='*50}")
summary_df = pd.DataFrame(results).T
print(summary_df.round(3))
summary_df.to_csv("/home/claude/nids/model_comparison.csv")
