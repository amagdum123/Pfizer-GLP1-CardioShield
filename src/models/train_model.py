import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

import joblib


# Load features
df = pd.read_csv("data/herg_features.csv")

print("Dataset:")
print(df.shape)


# Split features and target
X = df.drop("Y", axis=1)
y = df["Y"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


print("\nTraining model...")
model.fit(X_train, y_train)


# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]


# Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, prob))

print("\nClassification Report:")
print(classification_report(y_test, pred))


# Save model
joblib.dump(
    model,
    "models/random_forest_herg.pkl"
)

print("\nSaved model!")
