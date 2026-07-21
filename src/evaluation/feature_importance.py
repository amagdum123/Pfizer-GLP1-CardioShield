import pandas as pd
import joblib
import matplotlib.pyplot as plt


#load dataset that's combined
data = pd.read_csv("data/herg_features_combined.csv")

X = data.drop("Y", axis=1)


# Load model and get improtance values
model = joblib.load(
    "models/random_forest_combined_herg.pkl"
)
importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
})


#sort
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)
print("\nTop 20 Important Features:")
print(importance_df.head(20))


#save table to feature importance
importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)


# top 20 most important features
top = importance_df.head(20)
plt.figure(figsize=(10,6))
plt.barh(
    top["feature"],
    top["importance"]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features Influencing hERG Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(
    "results/feature_importance.png",
    dpi=300
)
print("\nSaved feature importance!")