import gradio as gr
import pandas as pd

rfm = pd.read_csv("/Users/darshanv/clv-prediction/data/rfm_with_predictions.csv")
orders = pd.read_csv("/Users/darshanv/clv-prediction/data/past_orders_processed.csv")
orders["InvoiceDate"] = pd.to_datetime(orders["InvoiceDate"])

customer_ids = sorted(rfm["Customer ID"].astype(str).unique())

def predict_clv(customer_id):
    customer_id = int(customer_id)
    row = rfm[rfm["Customer ID"] == customer_id].iloc[0]
    hist = orders[orders["Customer ID"] == customer_id].sort_values("InvoiceDate")

    summary = f"""
    Recency: {int(row['Recency'])} days
    Frequency: {int(row['Frequency'])} orders
    Total Past Spend: £{row['Monetary']:.2f}

    Predicted (next 90 days): £{row['PredictedSpend']:.2f}
    Actual: £{row['FutureSpend']:.2f}
    Error: £{abs(row['PredictedSpend'] - row['FutureSpend']):.2f}
    """
    chart_data = hist[["InvoiceDate", "OrderValue"]].copy()
    chart_data["InvoiceDate"] = pd.to_datetime(chart_data["InvoiceDate"]).dt.strftime("%Y-%m-%d")
    return summary, chart_data

demo = gr.Interface(
    fn=predict_clv,
    inputs=gr.Dropdown(choices=customer_ids, label="Select Customer ID"),
    outputs=[gr.Textbox(label="Prediction Summary"), gr.LinePlot(x="InvoiceDate", y="OrderValue")],
    title="Customer Lifetime Value Predictor",
    flagging_mode="never"
)

demo.launch()