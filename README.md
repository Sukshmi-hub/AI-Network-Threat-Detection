# AI-based Network Intrusion Detection System

Trains and evaluates ML models to classify network connections as normal
or one of four attack types (DoS, Probe, R2L, U2R), using the NSL-KDD
feature schema (41 features per connection).

## Pipeline
Run in order:
1. `01_generate_data.py` — loads the dataset. Currently generates a
   synthetic dataset matching the real NSL-KDD schema. **To use your real
   NSL-KDD/KDDCup99 files:** replace the `generate_synthetic_data()` call
   with `pd.read_csv("your_file.txt", names=COLUMN_NAMES)`.
2. `02_preprocess.py` — one-hot encodes categorical features (protocol,
   service, flag), scales numeric features, and does a stratified
   train/test split so rare attack types are represented in both sets.
3. `03_train.py` — trains a Random Forest and a neural network (MLP),
   using `class_weight='balanced'` on the Random Forest to counter class
   imbalance (attacks like U2R are ~1% of the data).
4. `04_evaluate.py` — computes accuracy/precision/recall/F1 and per-class
   breakdowns, and saves confusion matrix plots.

## Honest notes on the current version
- **Data**: synthetic, generated to match NSL-KDD's real column schema and
  class imbalance, so the pipeline is testable today. Swap in real NSL-KDD
  data (linked in this repo) using the one-line change noted above.
- **Neural net**: uses Scikit-learn's `MLPClassifier`, not TensorFlow, since
  TensorFlow wasn't installable in the build environment. Same core idea
  (feedforward neural network), different library. Worth porting to
  TensorFlow's `Sequential` API if the resume specifically claims
  TensorFlow.
- **Class imbalance**: uses `class_weight='balanced'`, not SMOTE, since
  `imbalanced-learn` wasn't available. This is a legitimate, simpler
  alternative — SMOTE can be added later with `pip install imbalanced-learn`.

## Result on synthetic data (illustrative)
Random Forest hit 84.8% overall accuracy, but recall on the `probe`
attack class was 0% — every probe attack was misclassified, mostly as
`dos`, because the two classes share similar traffic-volume features in
this dataset. This is a concrete example of why **accuracy alone is
misleading** for imbalanced security data, and why per-class
precision/recall/F1 and confusion matrices matter more.

## Next steps
- Swap in real NSL-KDD/UNSW-NB15 data
- Try SMOTE vs. class_weight and compare
- Port the neural net to TensorFlow if that's specifically claimed
- Investigate probe vs. dos feature overlap to improve probe recall
