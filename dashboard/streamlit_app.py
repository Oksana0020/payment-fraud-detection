from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data import ensure_dataset
from app.model import FEATURES, load_model, prepare_frame, score_transactions

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="shield",
    layout="wide",
)

st.title("Real-Time Payment Fraud Detection System")
st.caption(
    "Analyst dashboard for suspicious payment detection, risk scoring, "
    "and fraud prevention triage."
)

load_model()
dataset = ensure_dataset()
dataset_records = dataset[FEATURES + ["transaction_id"]].to_dict(
    orient="records"
)
scored_dataset = score_transactions(prepare_frame(dataset_records))
view = pd.concat(
    [
        dataset[["amount", "is_fraud"]].reset_index(drop=True),
        scored_dataset.reset_index(drop=True),
    ],
    axis=1,
)

col1, col2, col3 = st.columns(3)
col1.metric("Transactions", len(view))
col2.metric("Flagged Payments", int(view["is_suspicious"].sum()))
col3.metric("Average Risk Score", int(view["risk_score"].mean()))

st.subheader("Flagged transaction queue")
threshold = st.slider(
    "Minimum risk score",
    min_value=0,
    max_value=100,
    value=72,
)
filtered = view[view["risk_score"] >= threshold].sort_values(
    "risk_score",
    ascending=False,
)
st.dataframe(filtered.head(100), use_container_width=True)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig_risk = px.histogram(
        view,
        x="risk_score",
        nbins=20,
        title="Risk score distribution",
    )
    st.plotly_chart(fig_risk, use_container_width=True)
with chart_col2:
    fig_amount = px.scatter(
        view,
        x="amount",
        y="risk_score",
        color="is_suspicious",
        title="Payment amount vs. risk score",
        hover_data=["transaction_id"],
    )
    st.plotly_chart(fig_amount, use_container_width=True)

st.subheader("Analyze custom transactions")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    missing_features = [
        feature
        for feature in FEATURES + ["transaction_id"]
        if feature not in uploaded_df.columns
    ]
    if missing_features:
        st.error(f"Missing columns: {', '.join(missing_features)}")
    else:
        results = score_transactions(
            prepare_frame(uploaded_df.to_dict(orient="records"))
        )
        st.dataframe(results, use_container_width=True)
else:
    st.info(
        "Use the sample dataset in data/transactions_sample.csv or upload "
        "your own transaction export."
    )
