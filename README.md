# Customer Lifetime Value (CLV) Prediction

Predicts a customer's expected spend over the next 90 days based on their historical order behavior, using real transaction data from a UK-based online retailer (2009–2011).

## Business Problem
Businesses need to know which customers are worth extra marketing spend, retention offers, or attention. This project predicts each customer's future 90-day spend from their past purchase history, enabling data-driven customer prioritization — the same underlying problem companies like Amazon, Netflix, and Spotify solve at scale.

## Dataset
**Online Retail II** (UCI / Kaggle) — ~1M line-item transaction records, rolled up to ~37K orders across ~5,900 customers. After filtering to repeat customers (≥2 orders) with a valid time-based train/predict split, the final modeling set covers **4,023 customers**.

## Approach
Three modeling approaches were built and rigorously compared:

1. **Classical baseline — RFM (Recency, Frequency, Monetary) features + LightGBM**
2. **Deep learning — LSTM** on padded per-customer order sequences (up to 20 orders, features: days-since-last-order, order value, item count, cumulative spend, spend-relative-to-average)
3. **Hybrid model** — LSTM sequence representation concatenated with RFM features before a final prediction layer

All models predict `log1p(future_spend)` to handle a heavily right-skewed target, with predictions reversed via `expm1` for evaluation in £ terms.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **RFM + LightGBM (final choice)** | £612.80 | £3,547.15 | **0.3145** |
| Hybrid LSTM + RFM (avg. across 5 seeds) | £619.31 | £3,799.71 | 0.2102 |

The classical RFM + LightGBM baseline outperformed the deep learning approach on average, evaluated rigorously across 5 random seeds to control for neural network initialization variance — a finding consistent with published CLV literature at this dataset scale (~4,000 customers).

The full 11-iteration experiment log — including architecture changes, a checkpoint-saving bug found and fixed (missing `deepcopy`), and the discovery that an early promising result wasn't reproducible without a fixed random seed — is documented in [`reports/experiment_log.md`](reports/experiment_log.md).

## Key Engineering Decisions
- **Time-based train/predict split** (not random) to avoid data leakage — all features use only pre-cutoff history
- **Log-transformed target** to handle heavy right-skew in customer spend
- **Multi-seed evaluation** for the neural network, after discovering single-run results can be misleading due to random weight initialization
- **Precomputed predictions** for deployment, avoiding repeated native-library (LightGBM) calls inside the web app's request cycle

## Demo
An interactive dashboard (built with Gradio) lets you select any customer and view their order history, predicted vs. actual 90-day spend, and prediction error. Currently runs locally (see below) — no hosted public link at this time.

**Run locally:**
```bash
pip install -r requirements.txt
python app/app_gradio.py
```
This starts a local server (typically `http://127.0.0.1:7860`) — open that URL in your browser.

## Tech Stack
Python, pandas, numpy, scikit-learn, LightGBM, PyTorch (LSTM), Gradio

## Project Structure
```
clv-prediction/
├── data/               # raw + processed datasets
├── models/             # saved model artifacts
├── notebooks/          # data exploration + training scripts
├── reports/            # experiment log
├── app/                # Gradio dashboard
└── README.md
```