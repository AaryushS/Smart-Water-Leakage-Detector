def create_features(data):

    df = data.copy()

    df["flow_rolling_mean"] = (
        df["flow"]
        .rolling(window=6, min_periods=1)
        .mean()
    )

    df["pressure_rolling_mean"] = (
        df["pressure"]
        .rolling(window=6, min_periods=1)
        .mean()
    )

    df["flow_change"] = df["flow"].diff().fillna(0)

    df["pressure_change"] = (
        df["pressure"]
        .diff()
        .fillna(0)
    )

    previous_flow = (
        df["flow"]
        .shift(1)
        .replace(0, 1)
    )

    previous_pressure = (
        df["pressure"]
        .shift(1)
        .replace(0, 1)
    )

    df["flow_change_ratio"] = (
        df["flow_change"] /
        previous_flow
    )

    df["pressure_change_ratio"] = (
        df["pressure_change"] /
        previous_pressure
    )

    df["hour"] = df["timestamp"].dt.hour

    df["is_night"] = (
        df["hour"].between(0, 4)
        .astype(int)
    )

    return df