import numpy as np
import pandas as pd

from config import (
    NUMBER_OF_READINGS,
    NORMAL_FLOW_MIN,
    NORMAL_FLOW_MAX,
    NORMAL_PRESSURE_MIN,
    NORMAL_PRESSURE_MAX,
    ZONES,
    RANDOM_SEED
)


def generate_normal_data(number_of_readings=None):

    if number_of_readings is None:
        number_of_readings = NUMBER_OF_READINGS

    np.random.seed(RANDOM_SEED)

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=number_of_readings,
        freq="15min"
    )

    flow = np.random.normal(
        5,
        0.8,
        number_of_readings
    )

    pressure = np.random.normal(
        52,
        2,
        number_of_readings
    )

    flow = np.clip(
        flow,
        NORMAL_FLOW_MIN,
        NORMAL_FLOW_MAX
    )

    pressure = np.clip(
        pressure,
        NORMAL_PRESSURE_MIN,
        NORMAL_PRESSURE_MAX
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "flow": flow,
        "pressure": pressure
    })

    # Assign zones
    df["zone"] = [
        ZONES[i % len(ZONES)]
        for i in range(number_of_readings)
    ]

    # Ground truth
    df["actual_leak"] = False

    return df


def generate_sudden_leak():

    df = generate_normal_data()

    leak_start = 100
    leak_end = 130

    np.random.seed(RANDOM_SEED + 1)

    df.loc[
        leak_start:leak_end,
        "flow"
    ] += np.random.uniform(
        8,
        14,
        leak_end - leak_start + 1
    )

    df.loc[
        leak_start:leak_end,
        "pressure"
    ] -= np.random.uniform(
        5,
        10,
        leak_end - leak_start + 1
    )

    df.loc[
        leak_start:leak_end,
        "actual_leak"
    ] = True

    # Put sudden leak mainly in Zone C
    df.loc[
        leak_start:leak_end,
        "zone"
    ] = "Zone C"

    return df


def generate_slow_leak():

    df = generate_normal_data()

    leak_start = 60
    leak_end = 160

    increase = np.linspace(
        0,
        8,
        leak_end - leak_start + 1
    )

    df.loc[
        leak_start:leak_end,
        "flow"
    ] += increase

    df.loc[
        leak_start:leak_end,
        "pressure"
    ] -= increase * 0.35

    df.loc[
        leak_start:leak_end,
        "actual_leak"
    ] = True

    df.loc[
        leak_start:leak_end,
        "zone"
    ] = "Zone B"

    return df


def generate_night_leak():

    df = generate_normal_data()

    night_mask = df["timestamp"].dt.hour.between(0, 4)

    np.random.seed(RANDOM_SEED + 2)

    df.loc[
        night_mask,
        "flow"
    ] += np.random.uniform(
        7,
        12,
        night_mask.sum()
    )

    df.loc[
        night_mask,
        "pressure"
    ] -= np.random.uniform(
        4,
        8,
        night_mask.sum()
    )

    df.loc[
        night_mask,
        "actual_leak"
    ] = True

    df.loc[
        night_mask,
        "zone"
    ] = "Zone D"

    return df


def generate_pressure_leak():

    df = generate_normal_data()

    leak_start = 90
    leak_end = 140

    np.random.seed(RANDOM_SEED + 3)

    df.loc[
        leak_start:leak_end,
        "flow"
    ] += np.random.uniform(
        4,
        7,
        leak_end - leak_start + 1
    )

    df.loc[
        leak_start:leak_end,
        "pressure"
    ] -= np.random.uniform(
        15,
        22,
        leak_end - leak_start + 1
    )

    df.loc[
        leak_start:leak_end,
        "actual_leak"
    ] = True

    df.loc[
        leak_start:leak_end,
        "zone"
    ] = "Zone E"

    return df


def generate_data(scenario):

    if scenario == "Normal":
        return generate_normal_data()

    elif scenario == "Sudden Leakage":
        return generate_sudden_leak()

    elif scenario == "Slow Leakage":
        return generate_slow_leak()

    elif scenario == "Night Leakage":
        return generate_night_leak()

    elif scenario == "Pressure Leakage":
        return generate_pressure_leak()

    else:
        return generate_normal_data()