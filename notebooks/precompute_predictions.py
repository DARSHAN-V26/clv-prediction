import pandas as pd
import numpy as np
import joblib

rfm = pd.read_csv("/Users/darshanv/clv-prediction/data/rfm_features.csv")
model = joblib.load("/Users/darshanv/clv-prediction/models/rfm_baseline.pkl")

feature_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]
X = rfm[feature_cols]

pred_log = model.predict(X)
rfm["PredictedSpend"] = np.expm1(pred_log)

rfm.to_csv("/Users/darshanv/clv-prediction/data/rfm_with_predictions.csv", index=False)
print("Saved predictions for all customers.")
print(rfm[["Customer ID", "FutureSpend", "PredictedSpend"]].head())
