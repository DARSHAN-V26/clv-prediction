import streamlit as st
import pandas as pd

st.set_page_config(page_title="CLV Prediction Dashboard", layout="centered")

st.title("Customer Lifetime Value Predictor")
st.caption("Predicts a customer's expected spend over the next 90 days, based on their order history.")

@st.cache_data
def load_data():
    rfm = pd.read_csv("/Users/darshanv/clv-prediction/data/rfm_with_predictions.csv")
    orders = pd.read_csv("/Users/darshanv/clv-prediction/data/past_orders_processed.csv")
    orders["InvoiceDate"] = pd.to_datetime(orders["InvoiceDate"])
    return rfm, orders

rfm_df, orders_df = load_data()

customer_ids = sorted(rfm_df["Customer ID"].unique())
selected_id = st.selectbox("Select a Customer ID", customer_ids)

customer_row = rfm_df[rfm_df["Customer ID"] == selected_id].iloc[0]
customer_orders = orders_df[orders_df["Customer ID"] == selected_id].sort_values("InvoiceDate")

predicted_spend = customer_row["PredictedSpend"]
actual_spend = customer_row["FutureSpend"]

col1, col2, col3 = st.columns(3)
col1.metric("Recency (days)", int(customer_row["Recency"]))
col2.metric("Frequency (orders)", int(customer_row["Frequency"]))
col3.metric("Total Past Spend (£)", f"{customer_row['Monetary']:.2f}")

st.subheader("Predicted vs. Actual Spend — Next 90 Days")
pred_col, actual_col, error_col = st.columns(3)
pred_col.metric("Predicted (RFM + LightGBM)", f"£{predicted_spend:.2f}")
actual_col.metric("Actual", f"£{actual_spend:.2f}")
error_col.metric("Error", f"£{abs(predicted_spend - actual_spend):.2f}")

st.subheader("Order History")
st.line_chart(customer_orders.set_index("InvoiceDate")["OrderValue"])

st.dataframe(
    customer_orders[["InvoiceDate", "OrderValue", "NumItems"]].reset_index(drop=True),
    width="stretch"
)

st.divider()
with st.expander("Model Comparison — Why LightGBM was chosen over the LSTM"):
    st.markdown("""
    Two modeling approaches were tested for this project:

    | Model | Mean MAE | Mean RMSE | Mean R² |
    |---|---|---|---|
    | **RFM + LightGBM (used above)** | £612.80 | £3547.15 | **0.3145** |
    | Hybrid LSTM + RFM (avg. across 5 seeds) | £619.31 | £3799.71 | 0.2102 |

    A deep learning (LSTM-based) sequence model was built and rigorously tested, including a
    hybrid architecture combining LSTM-learned sequence patterns with RFM features. Averaged
    across 5 random seeds, it did not reliably outperform the simpler RFM + LightGBM baseline
    at this dataset size (~4,000 customers) — a finding consistent with published CLV research.
    The full experiment log, including 11 iterations and two real bugs caught during development,
    is documented in the project README.
    """)