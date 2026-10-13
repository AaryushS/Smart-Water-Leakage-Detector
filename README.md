# AI Smart Water Leakage Detector 💧

## Overview

The AI Smart Water Leakage Detector is a simulation-based project that identifies unusual water-flow and pressure patterns that may indicate leakage.

The system uses the Isolation Forest machine learning algorithm to detect anomalies and rule-based conditions to classify possible leakage types. A Streamlit dashboard displays sensor readings, alerts, estimated water loss, zone monitoring, and AI evaluation metrics.

This project is inspired by water-loss monitoring concepts used in water utility management.

## Features

- **AI Anomaly Detection:** Uses Isolation Forest to identify unusual sensor readings.
- **Leakage Classification:** Classifies possible sudden, slow, nighttime, and pressure-related leakage patterns.
- **Simulated Sensor Data:** Generates water-flow and pressure readings without physical sensors or external datasets.
- **Nighttime Flow Analysis:** Compares nighttime flow with an expected baseline.
- **Estimated Water Loss:** Estimates excess water from simulated flow readings.
- **Zone-Based Monitoring:** Displays possible leakage activity across five simulated zones.
- **Leak Severity:** Assigns low, medium, or high severity using rule-based conditions.
- **Interactive Dashboard:** Visualizes flow, pressure, anomaly scores, and leak information.
- **AI Performance Evaluation:** Displays accuracy, precision, recall, and F1-score against simulated leak labels.
- **Historical Trends:** Compares summaries from simulation runs within the current Streamlit session.
- **CSV Export:** Allows users to download analyzed sensor data.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Isolation Forest
- Streamlit
- Plotly

## Project Structure

```text
AI-Smart-Water-Leakage-Detector/
├── app.py
├── simulator.py
├── detector.py
├── preprocessing.py
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

1. The simulator generates water-flow, pressure, timestamp, and zone data.
2. The preprocessing module creates features from the sensor readings.
3. Isolation Forest identifies unusual readings.
4. Rule-based conditions classify possible leakage patterns and severity.
5. The dashboard presents alerts, estimated water loss, zone summaries, and evaluation metrics.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Smart-Water-Leakage-Detector
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual repository URL.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The dashboard will open in your browser. If it does not, open the local URL shown in the terminal.

## Available Simulation Scenarios

- Normal
- Sudden Leakage
- Slow Leakage
- Night Leakage
- Pressure Leakage

Select a scenario in the sidebar and click **Run Simulation** to view the results.

## Evaluation

The dashboard calculates accuracy, precision, recall, and F1-score by comparing Isolation Forest anomaly predictions with the simulator's actual leak labels.

These results evaluate the model on simulated data and should not be interpreted as guaranteed performance on real water networks.

## Limitations

- Sensor readings are simulated rather than collected from physical devices.
- Zone assignments are illustrative and do not represent actual geographic locations.
- Water-loss values are estimates based on a simplified baseline.
- Leak classifications and severity levels use prototype rules and require further validation.
- Historical summaries are stored only in the current application session.

## Future Scope

- Integration with real flow meters and pressure sensors.
- Persistent database storage for historical readings.
- Geographic leak visualization using GIS tools.
- More robust leakage classification and evaluation on real-world data.
- Automated notifications for suspected leakage.

## Author

Developed as an academic project to demonstrate machine learning-based anomaly detection and simulated water monitoring.

## Disclaimer

This project is an education al prototype. Its results are not intended to replace professional water-network inspection or validated leak-detection systems.