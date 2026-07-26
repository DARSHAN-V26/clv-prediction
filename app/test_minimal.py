import streamlit as st
import pandas as pd

st.title("Test Filtered Selectbox")
rfm = pd.read_csv("/Users/darshanv/clv-prediction/data/rfm_with_predictions.csv")
customer_ids = sorted(rfm["Customer ID"].unique())

search = st.text_input("Type a Customer ID to search")
if search:
    filtered_ids = [cid for cid in customer_ids if search in str(cid)]
else:
    filtered_ids = customer_ids[:50]  # show only first 50 by default

selected_id = st.selectbox("Select a Customer ID", filtered_ids)
st.write(selected_id)
