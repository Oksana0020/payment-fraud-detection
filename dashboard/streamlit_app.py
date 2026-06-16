from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from app.data import ensure_dataset
from app.model import (
    FEATURES,
    classify_risk_band,
    load_model,
    prepare_frame,
    score_transactions,
)


def infer_alert_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["amount"] >= 1000:
        reasons.append("large payment value")
    if row["transactions_last_24h"] >= 10:
        reasons.append("velocity spike")
    if row["is_international"]:
        reasons.append("cross-border activity")
    if row["is_night_transaction"]:
        reasons.append("late-night payment")
    if not row["device_trusted"]:
        reasons.append("untrusted device")
    if row["merchant_risk_score"] >= 0.75:
        reasons.append("high-risk merchant")
    if row["payment_method_risk"] >= 0.70:
        reasons.append("payment method risk")
    if row["geo_distance_km"] >= 800:
        reasons.append("location mismatch")
    if row["ip_velocity"] >= 20:
        reasons.append("ip velocity anomaly")

    if not reasons:
        return "routine payment behavior"
    return ", ".join(reasons)


@st.cache_data(show_spinner=False)
def load_dashboard_view() -> pd.DataFrame:
    load_model()
    dataset = ensure_dataset()
    dataset_records = dataset[FEATURES + ["transaction_id"]].to_dict(
        orient="records"
    )
    scored_dataset = score_transactions(prepare_frame(dataset_records))
    view = scored_dataset.merge(
        dataset[["transaction_id", "is_fraud"]],
        on="transaction_id",
        how="left",
    )
    view["risk_band"] = view["risk_score"].apply(classify_risk_band)
    view["alert_reasons"] = view.apply(infer_alert_reasons, axis=1)
    return view


st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="shield",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(15, 118, 110, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at top right,
                rgba(245, 158, 11, 0.12),
                transparent 24%
            ),
            linear-gradient(180deg, #f6fbfb 0%, #ffffff 42%, #f4f7f8 100%);
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #123b4a 52%,
            #0f766e 100%
        );
        color: #f8fafc;
        box-shadow: 0 24px 48px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero-kicker {
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        opacity: 0.72;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.05;
        margin-bottom: 0.55rem;
    }
    .hero-copy {
        max-width: 760px;
        opacity: 0.88;
        line-height: 1.55;
        margin-bottom: 0;
    }
    .section-note {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: rgba(15, 118, 110, 0.08);
        color: #0f172a;
        border: 1px solid rgba(15, 118, 110, 0.12);
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Fraud Operations Console</div>
        <div class="hero-title">Real-Time Payment Fraud Detection System</div>
        <p class="hero-copy">
            Review suspicious payments, inspect risk signals, and simulate
            analyst decisions across a lightweight fraud operations workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

view = load_dashboard_view()
model_bundle = load_model()
flagged_rate = (view["is_suspicious"].mean() * 100)

st.sidebar.header("Screening Controls")
threshold = st.sidebar.slider(
    "Escalation threshold",
    min_value=0,
    max_value=100,
    value=72,
)
show_only_flagged = st.sidebar.toggle("Flagged only", value=True)
selected_bands = st.sidebar.multiselect(
    "Risk bands",
    options=["Critical", "High", "Medium", "Low"],
    default=["Critical", "High", "Medium", "Low"],
)
selected_categories = st.sidebar.multiselect(
    "Merchant categories",
    options=sorted(view["merchant_category"].unique().tolist()),
    default=sorted(view["merchant_category"].unique().tolist()),
)

filtered = view[
    (view["risk_score"] >= threshold)
    & (view["risk_band"].isin(selected_bands))
    & (view["merchant_category"].isin(selected_categories))
].copy()
if show_only_flagged:
    filtered = filtered[filtered["is_suspicious"]]

top_row = st.columns(3)
top_row[0].metric("Transactions", len(view))
top_row[1].metric("Flagged Payments", int(view["is_suspicious"].sum()))
top_row[2].metric("Average Risk Score", int(view["risk_score"].mean()))

second_row = st.columns(3)
second_row[0].metric(
    "Critical Alerts",
    int((view["risk_band"] == "Critical").sum()),
)
second_row[1].metric(
    "Cross-Border Share",
    f"{(view['is_international'].mean() * 100):.0f}%",
)
second_row[2].metric("Model ROC AUC", f"{model_bundle.roc_auc:.2f}")

third_row = st.columns(3)
third_row[0].metric("Flag Rate", f"{flagged_rate:.0f}%")
third_row[1].metric(
    "Top Risk Category",
    (
        view.groupby("merchant_category")["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    ),
)
third_row[2].metric(
    "Median Payment",
    f"${view['amount'].median():,.0f}",
)

st.markdown(
    """
    <div class="section-note">
        This console is optimized for analyst triage: filter the queue,
        review the strongest signals, and test how the model responds to
        suspicious payment scenarios.
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, queue_tab, lab_tab, upload_tab = st.tabs(
    ["Overview", "Flag Queue", "Scenario Lab", "Bulk Upload"]
)

with overview_tab:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_risk = px.histogram(
            view,
            x="risk_score",
            nbins=24,
            color="risk_band",
            title="Risk score distribution",
            color_discrete_map={
                "Critical": "#b91c1c",
                "High": "#ea580c",
                "Medium": "#d97706",
                "Low": "#0f766e",
            },
        )
        fig_risk.update_layout(bargap=0.08)
        st.plotly_chart(fig_risk, use_container_width=True)

    with chart_col2:
        fig_amount = px.scatter(
            view,
            x="amount",
            y="risk_score",
            color="risk_band",
            title="Payment amount vs. risk score",
            hover_data=["transaction_id", "alert_reasons"],
            color_discrete_map={
                "Critical": "#b91c1c",
                "High": "#ea580c",
                "Medium": "#d97706",
                "Low": "#0f766e",
            },
        )
        st.plotly_chart(fig_amount, use_container_width=True)

    breakdown_col1, breakdown_col2 = st.columns(2)
    with breakdown_col1:
        category_risk = (
            view.groupby("merchant_category", as_index=False)["risk_score"]
            .mean()
            .sort_values("risk_score", ascending=False)
        )
        fig_category = px.bar(
            category_risk,
            x="merchant_category",
            y="risk_score",
            title="Average risk by merchant category",
            color="risk_score",
            color_continuous_scale=["#0f766e", "#f59e0b", "#b91c1c"],
        )
        st.plotly_chart(fig_category, use_container_width=True)

    with breakdown_col2:
        band_counts = (
            view["risk_band"]
            .value_counts()
            .rename_axis("risk_band")
            .reset_index(name="count")
        )
        fig_bands = px.pie(
            band_counts,
            names="risk_band",
            values="count",
            title="Queue composition by risk band",
            color="risk_band",
            color_discrete_map={
                "Critical": "#b91c1c",
                "High": "#ea580c",
                "Medium": "#d97706",
                "Low": "#0f766e",
            },
        )
        st.plotly_chart(fig_bands, use_container_width=True)

with queue_tab:
    st.subheader("Analyst queue")
    st.caption(
        "Escalation filters from the sidebar drive this queue. Inspect a "
        "single transaction before deciding whether to hold or approve it."
    )
    queue_view = filtered.sort_values(
        ["risk_score", "amount"],
        ascending=[False, False],
    )
    st.dataframe(
        queue_view[
            [
                "transaction_id",
                "risk_score",
                "risk_band",
                "amount",
                "merchant_category",
                "is_international",
                "transactions_last_24h",
                "recommended_action",
                "alert_reasons",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download current queue",
        data=queue_view.to_csv(index=False).encode("utf-8"),
        file_name="analyst_queue.csv",
        mime="text/csv",
    )

    if queue_view.empty:
        st.warning("No transactions match the current queue filters.")
    else:
        selected_id = st.selectbox(
            "Inspect transaction",
            options=queue_view["transaction_id"].tolist(),
        )
        selected_row = queue_view.loc[
            queue_view["transaction_id"] == selected_id
        ].iloc[0]

        detail_col1, detail_col2 = st.columns([1.3, 1])
        with detail_col1:
            st.markdown(
                f"**Recommended action:** {selected_row['recommended_action']}"
            )
            st.markdown(f"**Alert reasons:** {selected_row['alert_reasons']}")
            detail_frame = pd.DataFrame(
                {
                    "Field": [
                        "Amount",
                        "Risk score",
                        "Risk band",
                        "Merchant category",
                        "Cross-border",
                        "Night transaction",
                        "Trusted device",
                        "IP velocity",
                        "Geo distance (km)",
                    ],
                    "Value": [
                        f"${selected_row['amount']:.2f}",
                        int(selected_row["risk_score"]),
                        selected_row["risk_band"],
                        selected_row["merchant_category"],
                        bool(selected_row["is_international"]),
                        bool(selected_row["is_night_transaction"]),
                        bool(selected_row["device_trusted"]),
                        int(selected_row["ip_velocity"]),
                        float(selected_row["geo_distance_km"]),
                    ],
                }
            )
            st.dataframe(
                detail_frame,
                use_container_width=True,
                hide_index=True,
            )

        with detail_col2:
            st.metric(
                "Fraud probability",
                f"{selected_row['fraud_probability']:.2%}",
            )
            st.metric("Risk score", int(selected_row["risk_score"]))
            st.metric("Risk band", selected_row["risk_band"])
            st.metric(
                "Known label",
                "Fraud" if selected_row["is_fraud"] else "Legitimate",
            )

with lab_tab:
    st.subheader("Scenario lab")
    st.caption(
        "Simulate a transaction and inspect the model response before you "
        "turn the project into a stronger demo or portfolio case study."
    )
    with st.form("scenario_form"):
        form_col1, form_col2, form_col3 = st.columns(3)
        with form_col1:
            amount = st.number_input("Amount", min_value=1.0, value=1450.0)
            account_age_days = st.number_input(
                "Account age days",
                min_value=0,
                value=28,
            )
            transactions_last_24h = st.number_input(
                "Transactions last 24h",
                min_value=0,
                value=11,
            )
            avg_amount_7d = st.number_input(
                "Average amount 7d",
                min_value=0.0,
                value=210.0,
            )
        with form_col2:
            chargeback_history = st.number_input(
                "Chargeback history",
                min_value=0,
                value=1,
            )
            merchant_risk_score = st.slider(
                "Merchant risk score",
                min_value=0.0,
                max_value=1.0,
                value=0.82,
            )
            ip_velocity = st.number_input(
                "IP velocity",
                min_value=0,
                value=24,
            )
            geo_distance_km = st.number_input(
                "Geo distance km",
                min_value=0.0,
                value=1120.0,
            )
        with form_col3:
            payment_method_risk = st.slider(
                "Payment method risk",
                min_value=0.0,
                max_value=1.0,
                value=0.76,
            )
            customer_tenure_segment = st.selectbox(
                "Customer tenure segment",
                options=["new", "growing", "established", "vip"],
            )
            merchant_category = st.selectbox(
                "Merchant category",
                options=sorted(view["merchant_category"].unique().tolist()),
            )
            is_international = st.checkbox("International payment", value=True)
            is_night_transaction = st.checkbox("Night transaction", value=True)
            device_trusted = st.checkbox("Trusted device", value=False)

        submitted = st.form_submit_button("Score scenario")

    if submitted:
        scenario_row = {
            "transaction_id": "SIM-0001",
            "amount": amount,
            "account_age_days": int(account_age_days),
            "transactions_last_24h": int(transactions_last_24h),
            "avg_amount_7d": avg_amount_7d,
            "chargeback_history": int(chargeback_history),
            "is_international": is_international,
            "is_night_transaction": is_night_transaction,
            "device_trusted": device_trusted,
            "merchant_risk_score": merchant_risk_score,
            "ip_velocity": int(ip_velocity),
            "geo_distance_km": geo_distance_km,
            "payment_method_risk": payment_method_risk,
            "customer_tenure_segment": customer_tenure_segment,
            "merchant_category": merchant_category,
        }
        scenario_result = score_transactions(
            prepare_frame([scenario_row])
        ).iloc[0]
        st.success(
            f"Scenario scored at {int(scenario_result['risk_score'])} "
            f"({classify_risk_band(int(scenario_result['risk_score']))})."
        )
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric(
            "Fraud probability",
            f"{scenario_result['fraud_probability']:.2%}",
        )
        result_col2.metric("Risk score", int(scenario_result["risk_score"]))
        result_col3.metric(
            "Recommended action",
            scenario_result["recommended_action"],
        )

with upload_tab:
    st.subheader("Bulk upload review")
    uploaded_file = st.file_uploader(
        "Upload CSV for batch scoring",
        type="csv",
    )
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
            results["risk_band"] = results["risk_score"].apply(
                classify_risk_band
            )
            results["alert_reasons"] = results.apply(
                infer_alert_reasons,
                axis=1,
            )
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download scored results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="scored_transactions.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "Use the sample dataset in data/transactions_sample.csv or upload "
            "your own transaction export."
        )
