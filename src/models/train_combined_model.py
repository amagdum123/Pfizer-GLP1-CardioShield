import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# Load datasets

desc = pd.read_csv("data/herg_features.csv")
morgan = pd.read_csv("data/herg_morgan_features.csv")


# Separate labels

y = desc["Y"]

desc = desc.drop("Y", axis=1)
morgan = morgan.drop("Y", axis=1)


# Combine features

# Combine features

X = pd.concat(
    [desc, morgan],
    axis=1
)

# Add label back temporarily so we can save the full dataset
combined_df = X.copy()
combined_df["Y"] = y

combined_df.to_csv(
    "data/herg_features_combined.csv",
    index=False
)

print("Saved combined dataset!")

print("Combined dataset:")
print(X.shape)


# Split

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


# Train

print("\nTraining combined Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
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
    "models/random_forest_combined_herg.pkl"
)


print("\nSaved combined model!")