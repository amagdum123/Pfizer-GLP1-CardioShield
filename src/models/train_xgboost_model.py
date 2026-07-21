import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier


# Load combined dataset

df = pd.read_csv(
    "data/herg_features_combined.csv"
)

print("Dataset:")
print(df.shape)


# Split features and labels

X = df.drop("Y", axis=1)
y = df["Y"]


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


# Train XGBoost

print("\nTraining XGBoost...")


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
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


# Save model

joblib.dump(
    model,
    "models/xgboost_combined_herg.pkl"
)


print("\nSaved XGBoost model!")