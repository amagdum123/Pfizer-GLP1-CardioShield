import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import joblib


# Load Morgan fingerprints
df = pd.read_csv("data/herg_morgan_features.csv")

print("Dataset:")
print(df.shape)


X = df.drop("Y", axis=1)
y = df["Y"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training:")
print(X_train.shape)

print("Testing:")
print(X_test.shape)


# Train model
print("\nTraining Morgan Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# Evaluate

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]


print("\nAccuracy:")
print(accuracy_score(y_test, pred))


print("\nROC-AUC:")
print(roc_auc_score(y_test, prob))


print("\nClassification Report:")
print(classification_report(y_test, pred))


# Save

joblib.dump(
    model,
    "models/random_forest_morgan_herg.pkl"
)

print("\nSaved Morgan model!")