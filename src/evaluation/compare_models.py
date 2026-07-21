import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


#loading datasets
desc = pd.read_csv("data/herg_features.csv")
morgan = pd.read_csv("data/herg_morgan_features.csv")
combined = pd.read_csv("data/herg_features_combined.csv")


#evaluate model
def evaluate_model(name, X, y, model):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(
        y_test,
        probs
    )
    fpr, tpr, _ = roc_curve(
        y_test,
        probs
    )
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )
    return auc



# prep datasets
y = desc["Y"]
desc_X = desc.drop("Y", axis=1)
morgan_X = morgan.drop("Y", axis=1)
combined_X = combined.drop("Y", axis=1)
results = {}

#models
results["Descriptor RF"] = evaluate_model(
    "Descriptor RF",
    desc_X,
    y,
    RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
)
results["Morgan RF"] = evaluate_model(
    "Morgan RF",
    morgan_X,
    y,
    RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
)
results["Combined RF"] = evaluate_model(
    "Combined RF",
    combined_X,
    y,
    RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
)
results["XGBoost"] = evaluate_model(
    "XGBoost",
    combined_X,
    y,
    XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
)
#plot
plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "hERG Toxicity Model Comparison"
)
plt.legend()
plt.savefig(
    "results/roc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
#saving results to model comparision csv
pd.DataFrame(
    results.items(),
    columns=["Model","ROC-AUC"]
).to_csv(
    "results/model_comparison.csv",
    index=False
)
print("\nResults:")
print(results)
print("\nSaved ROC curve and comparison table")