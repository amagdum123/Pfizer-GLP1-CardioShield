import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

#load comabined data 
data = pd.read_csv(
    "data/herg_features_combined.csv"
)
X = data.drop("Y", axis=1)
y = data["Y"]
print("Dataset:")
print(X.shape)


#splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#random forest
param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"]
}
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)
print("\nStarting hyperparameter search...")
grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)
grid.fit(
    X_train,
    y_train
)
print("\nBest parameters:")
print(grid.best_params_)
print("\nBest CV ROC-AUC:")
print(grid.best_score_)


# Evaluate on test set
best_model = grid.best_estimator_
pred_prob = best_model.predict_proba(X_test)[:,1]
pred = best_model.predict(X_test)
print("\nTest Accuracy:")
print(accuracy_score(y_test, pred))
print("\nTest ROC-AUC:")
print(roc_auc_score(y_test, pred_prob))


#save new and improved model
joblib.dump(
    best_model,
    "models/random_forest_tuned_herg.pkl"
)
print("\nSaved tuned model!")