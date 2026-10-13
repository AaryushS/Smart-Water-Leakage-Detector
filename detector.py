import numpy as np

from sklearn.ensemble import IsolationForest

from simulator import generate_normal_data
from preprocessing import create_features


def train_normal_model():

    normal_data = generate_normal_data(
        number_of_readings=1000
    )

    normal_data = create_features(
        normal_data
    )

    features = [
        "flow",
        "pressure",
        "flow_change",
        "pressure_change",
        "flow_change_ratio",
        "pressure_change_ratio"
    ]

    X = normal_data[features].replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    return model, features


def detect_anomalies(data):

    df = create_features(
        data.copy()
    )

    model, features = train_normal_model()

    X = df[features].replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    predictions = model.predict(X)

    df["ml_anomaly"] = (
        predictions == -1
    )

    df["anomaly_score"] = (
        -model.decision_function(X)
    )

    return df


def classify_leakage(df):

    df = df.copy()

    # -------------------------
    # Sudden leakage
    # -------------------------

    sudden_condition = (
        (df["flow_change_ratio"] > 0.35)
        &
        (df["flow"] > 9)
    )

    # -------------------------
    # Slow leakage
    # -------------------------

    slow_condition = (
        df["flow"]
        .rolling(
            window=8,
            min_periods=1
        )
        .mean()
        > 6.5
    )

    # -------------------------
    # Night leakage
    # -------------------------

    night_condition = (
        (df["is_night"] == 1)
        &
        (df["flow"] > 7)
    )

    # -------------------------
    # Pressure leakage
    # -------------------------

    pressure_condition = (
        (df["pressure"] < 38)
        &
        (df["flow"] > 7)
    )

    # Default
    df["leakage_type"] = "Normal"

    anomaly = df["ml_anomaly"]

    # Slow
    df.loc[
        anomaly & slow_condition,
        "leakage_type"
    ] = "Slow Leakage"

    # Night
    df.loc[
        anomaly & night_condition,
        "leakage_type"
    ] = "Night Leakage"

    # Pressure
    df.loc[
        anomaly & pressure_condition,
        "leakage_type"
    ] = "Pressure Leakage"

    # Sudden
    df.loc[
        anomaly & sudden_condition,
        "leakage_type"
    ] = "Sudden Leakage"

    leakage_types = [
        "Sudden Leakage",
        "Pressure Leakage",
        "Night Leakage",
        "Slow Leakage"
    ]

    df["confirmed_leak"] = (
        df["leakage_type"]
        .isin(leakage_types)
    )

    # Unknown anomaly
    df.loc[
        anomaly & ~df["confirmed_leak"],
        "leakage_type"
    ] = "Unusual Usage"

    # -------------------------
    # Severity
    # -------------------------

    df["severity"] = "Low"

    medium_condition = (
        df["confirmed_leak"]
        &
        (
            (df["flow"] > 7)
            |
            (df["pressure"] < 45)
        )
    )

    high_condition = (
        df["confirmed_leak"]
        &
        (
            (df["flow"] > 10)
            |
            (df["pressure"] < 38)
            |
            (df["anomaly_score"] > 0.15)
        )
    )

    df.loc[
        medium_condition,
        "severity"
    ] = "Medium"

    df.loc[
        high_condition,
        "severity"
    ] = "High"

    return df