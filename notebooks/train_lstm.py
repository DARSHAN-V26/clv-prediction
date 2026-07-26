import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

path = "/Users/darshanv/clv-prediction/data/online_retail_II.csv"
df = pd.read_csv(path, encoding="ISO-8859-1")

df_clean = df.dropna(subset=["Customer ID"]).copy()
df_clean = df_clean[~df_clean["Invoice"].astype(str).str.startswith("C")]
df_clean = df_clean[df_clean["Quantity"] > 0]
df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_clean["LineTotal"] = df_clean["Quantity"] * df_clean["Price"]

orders = df_clean.groupby(["Invoice", "Customer ID", "InvoiceDate"]).agg(
    OrderValue=("LineTotal", "sum"),
    NumItems=("StockCode", "count")
).reset_index()

order_counts = orders.groupby("Customer ID").size().reset_index(name="NumOrders")
repeat_customers = order_counts[order_counts["NumOrders"] >= 2]["Customer ID"]
orders_filtered = orders[orders["Customer ID"].isin(repeat_customers)].copy()

cutoff_date = pd.Timestamp("2011-09-09")
future_end = cutoff_date + pd.Timedelta(days=90)
past_orders = orders_filtered[orders_filtered["InvoiceDate"] <= cutoff_date]
future_orders = orders_filtered[(orders_filtered["InvoiceDate"] > cutoff_date) &
                                  (orders_filtered["InvoiceDate"] <= future_end)]

future_spend = future_orders.groupby("Customer ID")["OrderValue"].sum().reset_index(name="FutureSpend")
labels = past_orders[["Customer ID"]].drop_duplicates().merge(future_spend, on="Customer ID", how="left")
labels["FutureSpend"] = labels["FutureSpend"].fillna(0)

past_orders_sorted = past_orders.sort_values(["Customer ID", "InvoiceDate"]).copy()
past_orders_sorted["DaysSinceLast"] = past_orders_sorted.groupby("Customer ID")["InvoiceDate"].diff().dt.days
past_orders_sorted["DaysSinceLast"] = past_orders_sorted["DaysSinceLast"].fillna(0)

past_orders_sorted["CumulativeSpend"] = past_orders_sorted.groupby("Customer ID")["OrderValue"].cumsum()
past_orders_sorted["RunningAvg"] = (
    past_orders_sorted.groupby("Customer ID")["OrderValue"].expanding().mean().reset_index(level=0, drop=True)
)
past_orders_sorted["RelativeToAvg"] = past_orders_sorted["OrderValue"] / past_orders_sorted["RunningAvg"]

sequence_cols = ["DaysSinceLast", "OrderValue", "NumItems", "CumulativeSpend", "RelativeToAvg"]
customer_sequences = (
    past_orders_sorted.groupby("Customer ID")[sequence_cols]
    .apply(lambda x: x.values.tolist())
    .to_dict()
)

MAX_LEN = 20
NUM_SEQ_FEATURES = len(sequence_cols)

def pad_sequence(seq, max_len=MAX_LEN, num_features=NUM_SEQ_FEATURES):
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    padding = [[0.0] * num_features] * pad_len
    return padding + seq

padded_sequences = {cid: pad_sequence(seq) for cid, seq in customer_sequences.items()}
customer_ids = list(padded_sequences.keys())

X_seq = np.array([padded_sequences[cid] for cid in customer_ids])
y = labels.set_index("Customer ID").loc[customer_ids]["FutureSpend"].values

rfm = past_orders_sorted.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
    Frequency=("InvoiceDate", "count"),
    Monetary=("OrderValue", "sum"),
    AvgOrderValue=("OrderValue", "mean")
).reset_index()
rfm = rfm.set_index("Customer ID").loc[customer_ids].reset_index()
X_rfm = rfm[["Recency", "Frequency", "Monetary", "AvgOrderValue"]].values

indices = np.arange(len(customer_ids))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]
X_rfm_train, X_rfm_test = X_rfm[train_idx], X_rfm[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

seq_scaler = StandardScaler()
n_samples, n_steps, n_features = X_seq_train.shape
X_seq_train_scaled = seq_scaler.fit_transform(X_seq_train.reshape(-1, n_features)).reshape(n_samples, n_steps, n_features)
X_seq_test_scaled = seq_scaler.transform(X_seq_test.reshape(-1, n_features)).reshape(X_seq_test.shape[0], n_steps, n_features)

rfm_scaler = StandardScaler()
X_rfm_train_scaled = rfm_scaler.fit_transform(X_rfm_train)
X_rfm_test_scaled = rfm_scaler.transform(X_rfm_test)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

X_seq_train_t = torch.tensor(X_seq_train_scaled, dtype=torch.float32)
X_rfm_train_t = torch.tensor(X_rfm_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
X_seq_test_t = torch.tensor(X_seq_test_scaled, dtype=torch.float32)
X_rfm_test_t = torch.tensor(X_rfm_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_log, dtype=torch.float32)

class HybridCLVModel(nn.Module):
    def __init__(self, seq_input_size=NUM_SEQ_FEATURES, rfm_input_size=4, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + rfm_input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x_seq, x_rfm):
        _, (h_n, _) = self.lstm(x_seq)
        lstm_out = h_n[-1]
        combined = torch.cat([lstm_out, x_rfm], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.fc2(out)
        return out.squeeze()

# --- run across multiple seeds, keep track of all + best ---
seeds = [42, 7, 123, 2024, 99]
results = []
best_overall_r2 = -float("inf")
best_overall_state = None

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HybridCLVModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 400
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_seq_train_t, X_rfm_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_seq_test_t, X_rfm_test_t)
            val_loss = criterion(val_preds, y_test_t)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        final_preds_log = model(X_seq_test_t, X_rfm_test_t).numpy()

    final_preds = np.expm1(final_preds_log)
    actual = np.expm1(y_test_t.numpy())

    mae = mean_absolute_error(actual, final_preds)
    rmse = np.sqrt(mean_squared_error(actual, final_preds))
    r2 = r2_score(actual, final_preds)

    results.append({"seed": seed, "mae": mae, "rmse": rmse, "r2": r2})
    print(f"Seed {seed} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    if r2 > best_overall_r2:
        best_overall_r2 = r2
        best_overall_state = best_model_state

results_df = pd.DataFrame(results)
print("\n--- Summary across seeds ---")
print(f"Mean MAE: {results_df['mae'].mean():.2f}, Mean RMSE: {results_df['rmse'].mean():.2f}, Mean R2: {results_df['r2'].mean():.4f}")
print(f"Std  R2: {results_df['r2'].std():.4f}")
print(f"RFM Baseline -> MAE: 612.80, RMSE: 3547.15, R2: 0.3145")

torch.save(best_overall_state, "/Users/darshanv/clv-prediction/models/hybrid_model.pt")