import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from simulator import generate_data
from detector import (
    detect_anomalies,
    classify_leakage
)


# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="AI Smart Water Leakage Detector",
    page_icon="💧",
    layout="wide"
)


# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("💧 AI Smart Water Leakage Detector")

st.write(
    "An AI-based simulated water monitoring system "
    "that detects abnormal water usage and identifies "
    "possible leakage patterns."
)


# ---------------------------------------
# SESSION STATE
# ---------------------------------------

if "data" not in st.session_state:
    st.session_state["data"] = None

if "history" not in st.session_state:
    st.session_state["history"] = []


# ---------------------------------------
# SIDEBAR
# ---------------------------------------

st.sidebar.header("Simulation Settings")

scenario = st.sidebar.selectbox(
    "Select Scenario",
    [
        "Normal",
        "Sudden Leakage",
        "Slow Leakage",
        "Night Leakage",
        "Pressure Leakage"
    ]
)

run_button = st.sidebar.button(
    "▶ Run Simulation",
    use_container_width=True
)


# ---------------------------------------
# RUN SIMULATION
# ---------------------------------------

if run_button:

    data = generate_data(
        scenario
    )

    data = detect_anomalies(
        data
    )

    data = classify_leakage(
        data
    )

    st.session_state["data"] = data
    st.session_state["scenario"] = scenario

    # -----------------------------------
    # History
    # -----------------------------------

    leak_count = int(
        data["confirmed_leak"].sum()
    )

    actual_leaks = int(
        data["actual_leak"].sum()
    )

    average_flow = float(
        data["flow"].mean()
    )

    history_record = {
        "Scenario": scenario,
        "Leak Points": leak_count,
        "Actual Leak Points": actual_leaks,
        "Average Flow": round(
            average_flow,
            2
        )
    }

    st.session_state["history"].append(
        history_record
    )


# ---------------------------------------
# CHECK DATA
# ---------------------------------------

if st.session_state["data"] is None:

    st.info(
        "Select a scenario from the sidebar "
        "and click Run Simulation."
    )

    st.stop()


df = st.session_state["data"]

current_scenario = st.session_state.get(
    "scenario",
    scenario
)


# =======================================
# BASIC CALCULATIONS
# =======================================

total_readings = len(df)

ai_anomalies = int(
    df["ml_anomaly"].sum()
)

actual_leaks = int(
    df["actual_leak"].sum()
)

ai_leaks = int(
    df["confirmed_leak"].sum()
)

average_flow = df["flow"].mean()


# =======================================
# WATER LOSS ESTIMATION
# =======================================

normal_data = generate_data(
    "Normal"
)

normal_night = normal_data[
    normal_data["timestamp"]
    .dt.hour
    .between(0, 4)
]

if len(normal_night) > 0:

    expected_night_flow = (
        normal_night["flow"].mean()
    )

else:

    expected_night_flow = 5


df["expected_flow"] = (
    df["timestamp"]
    .dt.hour
    .between(0, 4)
    .map({
        True: expected_night_flow,
        False: 5.0
    })
)


df["excess_flow"] = (
    df["flow"] -
    df["expected_flow"]
).clip(lower=0)


# Flow assumed to be L/min
# Each reading represents 15 minutes
df["estimated_water_loss_liters"] = (
    df["excess_flow"] * 15
)

total_water_loss = (
    df["estimated_water_loss_liters"]
    .sum()
)


# Water efficiency
total_actual_flow = (
    df["flow"].sum()
)

total_excess_flow = (
    df["excess_flow"].sum()
)

if total_actual_flow > 0:

    efficiency = (
        (total_actual_flow -
         total_excess_flow)
        /
        total_actual_flow
    ) * 100

else:

    efficiency = 100


efficiency = max(
    0,
    min(100, efficiency)
)


# =======================================
# TOP METRICS
# =======================================

st.subheader(
    f"📊 Current Scenario: {current_scenario}"
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Sensor Readings",
        total_readings
    )

with col2:

    st.metric(
        "AI Anomalies",
        ai_anomalies
    )

with col3:

    st.metric(
        "AI Detected Leaks",
        ai_leaks
    )

with col4:

    st.metric(
        "Estimated Water Loss",
        f"{total_water_loss:.0f} L"
    )

with col5:

    st.metric(
        "Water Efficiency",
        f"{efficiency:.1f}%"
    )


# =======================================
# ALERT
# =======================================

st.subheader("🚨 Leakage Alert")

if ai_leaks > 0:

    high_count = int(
        (
            df["severity"] == "High"
        ).sum()
    )

    if high_count > 0:

        st.error(
            f"🔴 HIGH PRIORITY: "
            f"{ai_leaks} possible leak points detected. "
            f"{high_count} require immediate inspection."
        )

    else:

        st.warning(
            f"🟡 Possible leakage detected at "
            f"{ai_leaks} points."
        )

else:

    st.success(
        "🟢 No confirmed leakage detected."
    )


# =======================================
# FLOW GRAPH
# =======================================

st.subheader("📈 Water Flow Monitoring")

# ---------------------------------------
# FLOW MONITORING GRAPH
# ---------------------------------------

st.subheader("Water Flow Monitoring")

fig_flow = px.line(
    df,
    x="timestamp",
    y="flow",
    title="Water Flow Over Time",
    labels={
        "timestamp": "Time",
        "flow": "Flow (L/min)"
    },
    template="plotly_white"
)

# Actual simulated leak points
actual_points = df[df["actual_leak"]]

if not actual_points.empty:
    fig_flow.add_scatter(
        x=actual_points["timestamp"],
        y=actual_points["flow"],
        mode="markers",
        name="Actual Leak",
        marker=dict(
            color="red",
            size=7,
            symbol="circle"
        )
    )

fig_flow.update_layout(
    height=420,
    hovermode="x unified",
    title_x=0.02,
    margin=dict(l=20, r=20, t=65, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#EAEAEA"
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#EAEAEA",
        rangemode="tozero"
    )
)

st.plotly_chart(
    fig_flow,
    use_container_width=True
)
# =======================================
# PRESSURE GRAPH
# =======================================

# ---------------------------------------
# PRESSURE MONITORING GRAPH
# ---------------------------------------

st.subheader("Water Pressure Monitoring")

fig_pressure = px.line(
    df,
    x="timestamp",
    y="pressure",
    title="Water Pressure Over Time",
    labels={
        "timestamp": "Time",
        "pressure": "Pressure"
    },
    template="plotly_white"
)

if not actual_points.empty:
    fig_pressure.add_scatter(
        x=actual_points["timestamp"],
        y=actual_points["pressure"],
        mode="markers",
        name="Actual Leak",
        marker=dict(
            color="red",
            size=7,
            symbol="circle"
        )
    )

fig_pressure.update_layout(
    height=380,
    hovermode="x unified",
    title_x=0.02,
    margin=dict(l=20, r=20, t=65, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#EAEAEA"
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#EAEAEA"
    )
)

st.plotly_chart(
    fig_pressure,
    use_container_width=True
)
# =======================================
# NIGHTTIME FLOW ANALYSIS
# =======================================

st.subheader("🌙 Nighttime Flow Analysis")

night_data = df[
    df["timestamp"]
    .dt.hour
    .between(0, 4)
]

if len(night_data) > 0:

    actual_night_flow = (
        night_data["flow"].mean()
    )

    excess_night_flow = max(
        0,
        actual_night_flow -
        expected_night_flow
    )

    night_loss = (
        night_data[
            "estimated_water_loss_liters"
        ].sum()
    )

else:

    actual_night_flow = 0
    excess_night_flow = 0
    night_loss = 0


n1, n2, n3, n4 = st.columns(4)

with n1:

    st.metric(
        "Expected Night Flow",
        f"{expected_night_flow:.2f} L/min"
    )

with n2:

    st.metric(
        "Actual Night Flow",
        f"{actual_night_flow:.2f} L/min"
    )

with n3:

    st.metric(
        "Excess Night Flow",
        f"{excess_night_flow:.2f} L/min"
    )

with n4:

    st.metric(
        "Estimated Night Loss",
        f"{night_loss:.0f} L"
    )


if excess_night_flow > 2:

    st.warning(
        "⚠️ Higher-than-normal nighttime flow "
        "may indicate hidden leakage."
    )

else:

    st.success(
        "🟢 Nighttime flow is within the expected range."
    )


# =======================================
# WATER LOSS
# =======================================

st.subheader("💧 Water Loss Analysis")

loss_col1, loss_col2 = st.columns(2)

with loss_col1:

    st.metric(
        "Estimated Excess Water",
        f"{total_water_loss:.0f} Liters"
    )

with loss_col2:

    st.metric(
        "Estimated Water Efficiency",
        f"{efficiency:.1f}%"
    )

st.caption(
    "Water loss is an estimate based on simulated "
    "flow above the expected baseline."
)


# =======================================
# ZONE MONITORING
# =======================================

st.subheader("📍 Zone-Based Leak Monitoring")

zone_summary = (
    df
    .groupby("zone")
    .agg(
        Sensor_Readings=("zone", "size"),
        AI_Anomalies=("ml_anomaly", "sum"),
        Confirmed_Leaks=("confirmed_leak", "sum"),
        Estimated_Water_Loss=("estimated_water_loss_liters", "sum")
    )
    .reset_index()
)

zone_summary[
    "Status"
] = "Normal"

zone_summary.loc[
    zone_summary["Confirmed_Leaks"] > 0,
    "Status"
] = "Suspicious"

zone_summary.loc[
    zone_summary["Confirmed_Leaks"] > 10,
    "Status"
] = "High Leakage"
    

st.dataframe(
    zone_summary,
    use_container_width=True,
    hide_index=True
)


# =======================================
# ZONE GRAPH
# =======================================

fig_zone = px.bar(
    zone_summary,
    x="zone",
    y="Confirmed_Leaks",
    title="Leaks Detected by Zone",
    labels={
        "zone": "Zone",
        "Confirmed_Leaks": "Detected Leaks"
    }
)

st.plotly_chart(
    fig_zone,
    use_container_width=True
)


# =======================================
# LEAK TYPE
# =======================================

st.subheader("🔍 Leakage Type Classification")

leak_type_counts = (
    df["leakage_type"]
    .value_counts()
    .reset_index()
)

leak_type_counts.columns = [
    "Leakage Type",
    "Count"
]

fig_type = px.bar(
    leak_type_counts,
    x="Leakage Type",
    y="Count",
    title="Leakage Types Detected"
)

st.plotly_chart(
    fig_type,
    use_container_width=True
)


# =======================================
# SEVERITY
# =======================================

st.subheader("⚠️ Leak Severity")

severity_counts = (
    df[
        df["confirmed_leak"] == True
    ]["severity"]
    .value_counts()
    .reset_index()
)

if len(severity_counts) > 0:

    severity_counts.columns = [
        "Severity",
        "Count"
    ]

    st.dataframe(
        severity_counts,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No confirmed leaks to classify."
    )


# =======================================
# ACTIVE LEAK LOCATIONS
# =======================================

st.subheader("🚨 Active Leak Locations")

active_leaks = df[
    df["confirmed_leak"] == True
]

if len(active_leaks) > 0:

    active_summary = (
        active_leaks
        .groupby(
            [
                "zone",
                "leakage_type",
                "severity"
            ]
        )
        .size()
        .reset_index(
            name="Leak Points"
        )
        .sort_values(
            "Leak Points",
            ascending=False
        )
    )

    st.dataframe(
        active_summary,
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No active leakage locations detected."
    )


# =======================================
# AI PERFORMANCE
# =======================================

st.subheader("🤖 AI Detection Performance")

y_true = (
    df["actual_leak"]
    .astype(int)
)

y_pred = (
    df["ml_anomaly"]
    .astype(int)
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.1f}%"
    )

with p2:

    st.metric(
        "Precision",
        f"{precision * 100:.1f}%"
    )

with p3:

    st.metric(
        "Recall",
        f"{recall * 100:.1f}%"
    )

with p4:

    st.metric(
        "F1 Score",
        f"{f1 * 100:.1f}%"
    )

st.caption(
    "Performance is calculated by comparing "
    "AI anomaly predictions with the simulated "
    "actual leak labels."
)


# =======================================
# ANOMALY SCORE
# =======================================


st.subheader("AI Anomaly Score")

fig_score = px.area(
    df,
    x="timestamp",
    y="anomaly_score",
    title="Unusual Activity Over Time",
    labels={
        "timestamp": "Time",
        "anomaly_score": "Anomaly Score"
    },
    template="plotly_white"
)

fig_score.update_traces(
    line=dict(width=2),
    fill="tozeroy"
)

fig_score.update_layout(
    height=350,
    hovermode="x unified",
    title_x=0.02,
    margin=dict(l=20, r=20, t=65, b=20),
    xaxis=dict(
        showgrid=True,
        gridcolor="#EAEAEA"
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#EAEAEA"
    )
)

st.plotly_chart(
    fig_score,
    use_container_width=True
)


# =======================================
# HISTORICAL TRENDS
# =======================================

st.subheader("📅 Historical Simulation Trends")

if len(st.session_state["history"]) > 0:

    history_df = pd.DataFrame(
        st.session_state["history"]
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    fig_history = px.bar(
        history_df,
        x="Scenario",
        y="Leak Points",
        title="Leak Points Across Simulations"
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )

else:

    st.info(
        "Run multiple simulations to build history."
    )


# =======================================
# SENSOR DATA
# =======================================

st.subheader("📋 Sensor Data")

display_columns = [
    "timestamp",
    "zone",
    "flow",
    "pressure",
    "ml_anomaly",
    "anomaly_score",
    "actual_leak",
    "confirmed_leak",
    "leakage_type",
    "severity"
]

st.dataframe(
    df[display_columns].tail(50),
    use_container_width=True,
    hide_index=True
)


# =======================================
# DOWNLOAD
# =======================================

csv = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Sensor Data CSV",
    data=csv,
    file_name="water_leak_analysis.csv",
    mime="text/csv",
    use_container_width=True
)


# =======================================
# FOOTER
# =======================================

st.markdown("---")

st.caption(
    "AI Smart Water Leakage Detector | "
    "Isolation Forest + Rule-Based Classification | "
    "Simulation-Based Water Monitoring"
)