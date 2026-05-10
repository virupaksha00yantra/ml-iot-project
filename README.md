# 🌿 Smart IoT Plant Monitoring System with Machine Learning-based Health Prediction

> **Academic Project | IoT + Machine Learning | Python Simulation**  
> A complete end-to-end simulation of an IoT plant health monitoring system using sensor data generation, Random Forest classification, and a live Streamlit web dashboard — no hardware required.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [IoT System Architecture](#4-iot-system-architecture)
5. [Technologies Used](#5-technologies-used)
6. [Project Structure](#6-project-structure)
7. [Implementation Process](#7-implementation-process)
8. [Machine Learning Methodology](#8-machine-learning-methodology)
9. [Installation and Setup](#9-installation-and-setup)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [Dashboard Features](#11-dashboard-features)
12. [Results and Discussion](#12-results-and-discussion)
13. [Challenges and How We Solved Them](#13-challenges-and-how-we-solved-them)
14. [Conclusion](#14-conclusion)
15. [Future Scope](#15-future-scope)
16. [References](#16-references)

---

## 1. Project Overview

| Field | Details |
|-------|---------|
| **Project Title** | Smart IoT Plant Monitoring System with ML-based Health Prediction |
| **Domain** | Internet of Things (IoT) + Machine Learning |
| **Type** | Complete Software Simulation (No Hardware Needed) |
| **Language** | Python 3.9+ |
| **ML Algorithm** | Random Forest Classifier |
| **Model Accuracy** | 99.58% Test Accuracy, 98.50% Cross-Validation |
| **Dashboard** | Streamlit Web Application (Real-time) |
| **Dataset** | 1,200 rows, 4 features, 4 health classes (self-generated) |

---

## 2. Problem Statement

### 2.1 The Real-World Problem

Plants — whether in homes, greenhouses, or farms — are extremely sensitive to their environment. Four key environmental factors determine whether a plant thrives or suffers:

- **Soil Moisture** — Too dry and roots die; too wet and roots rot.
- **Temperature** — Extreme heat causes wilting, enzyme breakdown, and irreversible stress.
- **Humidity** — Low humidity causes rapid water loss through leaves (transpiration stress).
- **Light Intensity** — Insufficient light stops photosynthesis; the plant slowly starves.

Most people and even farmers **react to plant problems only after visible damage appears** — yellowing leaves, wilting, or root rot. By that point, recovery is difficult and costly. There is a critical need for a system that can **predict plant health problems before they become visible**, based on real-time environmental data.

### 2.2 Existing Limitations

- Traditional monitoring systems only **display raw sensor values** without interpreting what they mean for the plant.
- Manual inspection is **time-consuming, error-prone**, and impossible at scale (e.g., large greenhouses).
- Hardware-based IoT deployments are **expensive and inaccessible** to many students and small farmers.
- There is no single unified system that **collects, interprets, and alerts** in real time from a simple laptop.

### 2.3 What This Project Solves

This project builds a system that:
1. Continuously **reads environmental sensor data** (simulated but statistically realistic).
2. **Automatically classifies** the plant's health into one of 4 conditions using ML.
3. **Sends real-time alerts** when the plant needs attention.
4. **Runs entirely on a laptop** with no physical hardware or sensors needed.

---

## 3. Proposed Solution

### 3.1 Core Idea

We designed a **4-layer IoT architecture** entirely in software. Instead of physical ESP32 microcontrollers and sensors, we use Python to simulate realistic sensor readings using statistical distributions calibrated to real-world sensor behaviour. A trained Machine Learning model then classifies the plant's health from these readings, and a live web dashboard displays everything in real time.

### 3.2 The 4 Plant Health Classes

The ML model classifies the plant into one of four states:

| Class | Label | What It Means | Key Sensor Signature |
|-------|-------|---------------|----------------------|
| 0 | ✅ **Healthy** | All conditions optimal | Soil ~60%, Temp ~24°C, Humidity ~60%, Light ~700 lux |
| 1 | 💧 **Needs Water** | Soil critically dry | Soil ~20%, other sensors near-normal |
| 2 | 🔥 **High Temperature Stress** | Overheating, dehydrating | Temp ~38°C, Humidity ~30% |
| 3 | 🌑 **Low Light Condition** | Insufficient photosynthesis | Light ~200 lux |

### 3.3 Solution Architecture (Summary)

```
Simulated IoT Sensors  -->  Data Logger (CSV)  -->  ML Model Training
                                                           |
        Live Dashboard  <--  Real-time Prediction  <--  Saved Model (.pkl)
```

---

## 4. IoT System Architecture

The project follows the standard **4-Layer IoT Reference Architecture**:

```
+========================================================================+
|                    LAYER 1: PERCEPTION LAYER                           |
|                      (IoT Sensor Nodes)                                |
|                                                                        |
|  [Soil Moisture]  [Temperature]   [Humidity]   [Light Intensity]       |
|   (0 - 100%)      (10 - 50 C)    (10 - 100%)   (0 - 1200 lux)        |
+================================|=======================================+
                                 |
                   Raw sensor readings
                                 |
+================================v=======================================+
|                    LAYER 2: NETWORK LAYER                              |
|                 (Data Ingestion and Transmission)                      |
|                                                                        |
|    sensor_simulation.py  ----------------->  dataset.csv              |
|    (Simulates MQTT edge gateway)              (IoT data log file)      |
+================================|=======================================+
                                 |
                   Structured CSV with timestamps
                                 |
+================================v=======================================+
|                   LAYER 3: PROCESSING LAYER                            |
|                     (Machine Learning Pipeline)                        |
|                                                                        |
|  dataset.csv -> Preprocessing -> Random Forest -> Evaluation -> .pkl   |
|                  (ml_model.py -- training pipeline)                    |
+================================|=======================================+
                                 |
                   Prediction + confidence score
                                 |
+================================v=======================================+
|                  LAYER 4: APPLICATION LAYER                            |
|                   (User-facing Dashboard and Alerts)                   |
|                                                                        |
|   app.py (Streamlit)  -->  Live Prediction  -->  Alerts + Graphs      |
+========================================================================+
```

### Layer-by-Layer Explanation

**Layer 1 — Perception Layer (Sensors)**

In a real deployment, this layer consists of physical IoT hardware:
- **Soil Moisture Sensor** (capacitive or resistive) connected to an ADC pin
- **DHT22 / BME280** for temperature and humidity
- **BH1750 or LDR** for light intensity in lux

In our simulation, `sensor_simulation.py` uses `numpy.random.normal(mean, std)` to generate readings that match the statistical distribution of real sensors, including natural noise and variation.

**Layer 2 — Network Layer (Communication)**

In a real IoT system, this layer handles wireless transmission using protocols like:
- **MQTT** (lightweight publish/subscribe protocol used in IoT)
- **Wi-Fi / LoRa / Zigbee** depending on range requirements
- **Cloud ingestion** via AWS IoT Core or Azure IoT Hub

In our simulation, `sensor_simulation.py` writes data directly to `dataset.csv`, mirroring what a real edge gateway would transmit and log.

**Layer 3 — Processing Layer (Intelligence)**

This is the brain of the system. `ml_model.py`:
- Loads the sensor data from CSV
- Preprocesses it (feature extraction, train/test split)
- Trains a **Random Forest Classifier** with 200 decision trees
- Evaluates the model (confusion matrix, cross-validation, classification report)
- Saves the trained model to `plant_model.pkl` for live use

**Layer 4 — Application Layer (Dashboard)**

`app.py` is the end-user interface built with Streamlit:
- Loads the saved model
- Continuously generates live sensor readings
- Runs the ML model on each new reading
- Displays health status, confidence scores, alerts, and historical graphs
- Provides a manual prediction tool via sidebar sliders

---

## 5. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.9+ | Core programming language |
| **NumPy** | >=1.23 | Sensor data generation using statistical distributions |
| **Pandas** | >=1.5 | Data manipulation, CSV reading/writing |
| **Scikit-learn** | >=1.2 | Random Forest training, evaluation, cross-validation |
| **Matplotlib** | >=3.6 | Plotting confusion matrix, feature importance, distributions |
| **Seaborn** | >=0.12 | Enhanced statistical visualisations (heatmaps) |
| **Streamlit** | >=1.28 | Live interactive web dashboard |
| **Pickle** | built-in | Model serialisation and loading |

### Why These Technologies?

- **NumPy random.normal()** produces sensor values with realistic Gaussian noise, matching real sensor behaviour
- **Scikit-learn Random Forest** provides excellent out-of-the-box accuracy for tabular IoT data, no feature scaling needed, fast inference
- **Streamlit** builds production-quality web dashboards in pure Python, no HTML/CSS/JavaScript needed
- **Pickle** is lightweight and fast for model serialisation, saving and loading the trained classifier

---

## 6. Project Structure

```
smart-plant-monitor/
|
|-- sensor_simulation.py    # Step 1: Simulate IoT sensors, generate dataset
|-- ml_model.py             # Step 2: Train ML model, evaluate, save
|-- app.py                  # Step 3: Streamlit live dashboard
|
|-- dataset.csv             # Auto-generated: 1,200 sensor readings
|-- plant_model.pkl         # Auto-generated: trained Random Forest model
|-- requirements.txt        # Python package dependencies
|
+-- plots/
    |-- confusion_matrix.png       # Actual vs predicted health classes
    |-- feature_importance.png     # Which sensor is most important
    |-- sensor_distributions.png   # Histogram of sensor values per class
    +-- cv_scores.png              # 5-fold cross-validation accuracy per fold
```

### File Responsibilities

| File | Responsibility |
|------|----------------|
| `sensor_simulation.py` | IoT layer simulation, dataset generation |
| `ml_model.py` | ML training pipeline, evaluation, plot generation |
| `app.py` | Full Streamlit dashboard, live prediction, alerts |
| `requirements.txt` | Dependency management |

---

## 7. Implementation Process

This section explains **how we built the project step by step** — from idea to working system.

### Step 1: Designing the Sensor Simulation

**The challenge:** We needed realistic sensor data without physical hardware.

**Our approach:** We studied real plant sensor data from agriculture research papers and IoT documentation to identify what values a healthy plant's sensors typically show, and how those values shift when the plant is stressed.

We then defined **Gaussian distribution profiles** for each sensor under each health condition:

```python
CLASS_PROFILES = {
    0: {  # Healthy
        "soil_moisture": (60, 8),    # mean=60%, std=8%
        "temperature":   (24, 3),    # mean=24 C, std=3
        "humidity":      (60, 8),    # mean=60%, std=8%
        "light":         (700, 100), # mean=700 lux, std=100
    },
    1: {  # Needs Water
        "soil_moisture": (20, 7),    # Low moisture -- key signal
        "temperature":   (26, 4),
        "humidity":      (45, 8),
        "light":         (650, 120),
    },
    # ... and so on for High Temp Stress and Low Light
}
```

We generated 300 samples per class using `numpy.random.normal()`, clipped values to realistic sensor bounds, shuffled the dataset, and saved it as `dataset.csv`.

**Why Gaussian distribution?** Real sensors produce readings that cluster around an optimal value with natural variation. Gaussian (normal) distribution perfectly models this behaviour — just like real sensors do.

---

### Step 2: Building the ML Training Pipeline

**The challenge:** Choosing the right algorithm and training a reliable model.

**Algorithm Selection:** We evaluated three options:

| Algorithm | Pros | Cons | Chosen? |
|-----------|------|------|---------|
| Decision Tree | Simple, interpretable | Overfits easily | No |
| **Random Forest** | Robust, accurate, feature importance built-in | Slightly slower | YES |
| SVM | Good for high-dimensional data | Needs feature scaling, slower | No |

Random Forest was the clear winner for tabular IoT sensor data.

**Training Process:**

```python
# 1. Load and split the data
X = df[["soil_moisture", "temperature", "humidity", "light_intensity"]].values
y = df["health_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y ensures equal class representation in both train and test sets

# 2. Train the model
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 3. Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # --> 99.58%

# 4. Cross-validate to verify generalisation
cv_scores = cross_val_score(clf, X, y, cv=5)  # --> 98.50% +/- 0.42%

# 5. Save the trained model
with open("plant_model.pkl", "wb") as f:
    pickle.dump(clf, f)
```

**Evaluation Plots Generated:**
- **Confusion Matrix** — shows which classes were misclassified and how many times
- **Feature Importance** — shows which sensor contributes most to predictions
- **Sensor Distributions** — histograms showing value ranges per class
- **CV Score Bar Chart** — shows per-fold accuracy across 5 folds

---

### Step 3: Building the Streamlit Dashboard

**The challenge:** Making the ML model accessible as a real-time monitoring interface.

**Architecture of app.py:**

```
On every page load or refresh:
  1. Load plant_model.pkl (cached with @st.cache_resource)
  2. Call simulate_reading() -- generates new sensor values
  3. Pass values to clf.predict() -- get health class
  4. Pass values to clf.predict_proba() -- get confidence %
  5. If class != Healthy --> show red alert banner
  6. Append reading to session history (last 50 kept)
  7. Render: metric cards + trend charts + pie chart + eval plots
```

**Live Prediction Code:**
```python
def simulate_reading():
    return {
        "soil_moisture":   round(random.gauss(55, 18), 1),
        "temperature":     round(random.gauss(28, 7), 1),
        "humidity":        round(random.gauss(55, 15), 1),
        "light_intensity": round(random.gauss(600, 200), 0),
    }

reading = simulate_reading()
X_now = np.array([[reading["soil_moisture"], reading["temperature"],
                   reading["humidity"],      reading["light_intensity"]]])

prediction = clf.predict(X_now)[0]              # --> 0, 1, 2, or 3
confidence = clf.predict_proba(X_now)[0].max()  # --> e.g. 0.97
```

**Session State for History:**
```python
# Streamlit session state persists data across re-runs within a session
if "history" not in st.session_state:
    st.session_state.history = []
st.session_state.history.append(reading)
# Keeps last 50 readings for the trend charts
```

---

### Step 4: Integration and Testing

After building all three scripts, we tested the full pipeline end to end:

1. Ran `sensor_simulation.py` and verified `dataset.csv` has 1,200 balanced rows
2. Ran `ml_model.py` and verified 99.58% accuracy and all 4 plot files generated
3. Ran `streamlit run app.py` and verified live predictions, alerts, and charts work correctly
4. Manually tested edge cases using the sidebar sliders:
   - Soil moisture = 10% → model correctly predicts "Needs Water"
   - Temperature = 42°C → model correctly predicts "High Temperature Stress"
   - Light = 80 lux → model correctly predicts "Low Light Condition"
   - All values optimal → model correctly predicts "Healthy"

All four manual tests passed with confidence scores above 90%.

---

## 8. Machine Learning Methodology

### 8.1 Dataset Details

| Property | Value |
|----------|-------|
| Total samples | 1,200 |
| Samples per class | 300 (balanced) |
| Input features | 4 (soil moisture, temperature, humidity, light) |
| Target variable | Health label (0, 1, 2, 3) |
| Random seed | 42 (reproducible) |
| Train/Test split | 80% / 20% (960 train, 240 test) |

### 8.2 Sensor Profiles Per Health Class

| Class | Soil Moisture | Temperature | Humidity | Light Intensity |
|-------|--------------|-------------|----------|-----------------|
| Healthy | mean=60%, std=8 | mean=24°C, std=3 | mean=60%, std=8 | mean=700 lux, std=100 |
| Needs Water | mean=20%, std=7 | mean=26°C, std=4 | mean=45%, std=8 | mean=650 lux, std=120 |
| High Temp Stress | mean=45%, std=8 | mean=38°C, std=4 | mean=30%, std=7 | mean=680 lux, std=100 |
| Low Light | mean=55%, std=8 | mean=22°C, std=3 | mean=58%, std=8 | mean=200 lux, std=80 |

### 8.3 Random Forest — How It Works

A Random Forest is an ensemble of many Decision Trees. Here is how it classifies plant health:

```
New sensor reading: [soil=18%, temp=26 C, humidity=43%, light=640 lux]
                              |
              +---------------+---------------+
              |               |               |
           Tree 1          Tree 2   ...   Tree 200
         "Needs Water"  "Needs Water"   "Needs Water"
              |               |               |
              +---------------+---------------+
                              |
                       Majority Vote
                              |
                  Final: "Needs Water" (confidence: 98%)
```

Each tree is trained on a random bootstrap sample of the training data and a random subset of features. This diversity among trees prevents overfitting and improves generalisation.

### 8.4 Model Hyperparameters

```python
RandomForestClassifier(
    n_estimators = 200,    # 200 decision trees in the forest
    max_depth    = None,   # Trees grow until all leaves are pure
    random_state = 42,     # Fixed seed for reproducibility
    n_jobs       = -1,     # Use all available CPU cores (faster training)
)
```

### 8.5 Evaluation Results

**Test Set Performance (240 samples):**

| Metric | Score |
|--------|-------|
| Test Accuracy | **99.58%** |
| 5-Fold CV Accuracy | **98.50% +/- 0.42%** |
| Macro Precision | 0.9959 |
| Macro Recall | 0.9958 |
| Macro F1-Score | 0.9958 |

**Per-Class Breakdown:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 1.0000 | 1.0000 | 1.0000 | 60 |
| Needs Water | 1.0000 | 0.9833 | 0.9916 | 60 |
| High Temp Stress | 0.9836 | 1.0000 | 0.9917 | 60 |
| Low Light | 1.0000 | 1.0000 | 1.0000 | 60 |

**Confusion Matrix (240 test samples):**

```
                    Predicted
                 Healthy  NeedsWater  HighTemp  LowLight
Actual Healthy  [  60         0          0         0  ]
    NeedsWater  [   0        59          1         0  ]
      HighTemp  [   0         0         60         0  ]
      LowLight  [   0         0          0        60  ]
```

Only 1 sample was misclassified out of 240.

**Feature Importance Ranking:**

| Rank | Feature | Importance | Reason |
|------|---------|------------|--------|
| 1 | `light_intensity` | ~0.42 | Low Light class uniquely distinct at ~200 lux |
| 2 | `temperature` | ~0.28 | High Temp class uniquely high at ~38°C |
| 3 | `soil_moisture` | ~0.18 | Needs Water class uniquely low at ~20% |
| 4 | `humidity` | ~0.12 | Least discriminating; overlaps across classes |

---

## 9. Installation and Setup

### 9.1 System Requirements

| Requirement | Minimum |
|-------------|---------|
| Operating System | Windows 10 / macOS 10.14 / Ubuntu 18.04 |
| Python | 3.9 or later |
| RAM | 4 GB (8 GB recommended) |
| Storage | ~200 MB (including Python packages) |
| Internet | Required only for initial package installation |

### 9.2 Check if Python is Installed

Open a terminal or command prompt and run:

```bash
python --version
```

You should see something like `Python 3.11.2`. If not, download Python from [python.org/downloads](https://www.python.org/downloads/).

> **Windows users:** During Python installation, make sure to check the box **"Add Python to PATH"**. This is very important.

### 9.3 Download the Project Files

Download these files and place them all in a single folder, for example `smart-plant-monitor/`:

- `sensor_simulation.py`
- `ml_model.py`
- `app.py`
- `requirements.txt`

### 9.4 Open a Terminal in the Project Folder

**Windows:**
1. Open the project folder in File Explorer
2. Click the address bar at the top, type `cmd`, press Enter

**Mac / Linux:**
1. Right-click the folder and select "Open Terminal here"
2. Or open Terminal and type: `cd path/to/your/folder`

### 9.5 Install All Required Packages

Run this single command:

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `streamlit`.

Expected output ends with:
```
Successfully installed streamlit-1.xx pandas-2.x scikit-learn-1.x ...
```

> If `pip` is not found, try `pip3` instead on Mac/Linux.

---

## 10. How to Run the Project

Run the three scripts **in order**. Each one builds on the output of the previous.

### Command 1 — Generate the Sensor Dataset

```bash
python sensor_simulation.py
```

**What it does:** Simulates 1,200 IoT sensor readings across 4 plant health conditions and saves them to `dataset.csv`.

**Expected terminal output:**
```
Generating dataset...
Saved 1200 rows to dataset.csv

Class distribution:
Healthy                    300
Needs Water                300
High Temperature Stress    300
Low Light Condition        300
```

**File created:** `dataset.csv`

---

### Command 2 — Train the Machine Learning Model

```bash
python ml_model.py
```

**What it does:** Loads `dataset.csv`, trains the Random Forest classifier, evaluates it, saves the model, and generates 4 evaluation plots inside the `plots/` folder.

**Expected terminal output:**
```
[1/5] Loading data ...       1200 rows, 4 features
[2/5] Splitting 80/20 ...
[3/5] Training Random Forest (200 trees) ...
[4/5] Evaluating ...

  Test Accuracy      : 99.58%
  CV Accuracy (5-fold): 98.50% +/- 0.42%

[5/5] Saving model and plots ...
  Model saved --> plant_model.pkl
  Saved plots/confusion_matrix.png
  Saved plots/feature_importance.png
  Saved plots/sensor_distributions.png
  Saved plots/cv_scores.png

Training complete!
```

**Files created:** `plant_model.pkl` and all 4 PNG charts inside `plots/`

---

### Command 3 — Launch the Live Dashboard

```bash
streamlit run app.py
```

**What it does:** Starts a local web server and opens the interactive dashboard in your browser automatically.

**Expected terminal output:**
```
  You can now view your Streamlit app in your browser.
  Local URL:   http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser will open automatically at `http://localhost:8501`. If it does not open, copy that URL and paste it into your browser manually.

To stop the dashboard, press `Ctrl + C` in the terminal.

---

### All 3 Commands — Quick Reference

```bash
python sensor_simulation.py
python ml_model.py
streamlit run app.py
```

---

### Troubleshooting Common Errors

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `python: command not found` | Python not installed or not in PATH | Reinstall Python, tick "Add to PATH" |
| `ModuleNotFoundError: No module named 'streamlit'` | Packages not installed | Run `pip install -r requirements.txt` |
| `FileNotFoundError: dataset.csv` | Skipped Command 1 | Run `python sensor_simulation.py` first |
| `FileNotFoundError: plant_model.pkl` | Skipped Command 2 | Run `python ml_model.py` first |
| `streamlit: command not found` | Streamlit not installed | Run `pip install streamlit` |
| Port 8501 already in use | Another Streamlit session running | Run `streamlit run app.py --server.port 8502` |

---

## 11. Dashboard Features

### Sidebar — Controls and Manual Prediction

- **Auto Refresh toggle** — When ON, the dashboard refreshes every 3 seconds, simulating live IoT data streaming.
- **Refresh Now button** — Manually trigger a new sensor reading instantly.
- **Manual Prediction sliders** — Drag sliders to set custom sensor values, then click "Predict" to test the model with your own inputs.

### Alert Banner

- **Green banner** — Plant is Healthy. All readings are in optimal range.
- **Red banner** — Plant needs attention. Displays the specific condition (e.g., "Needs Water") and the ML confidence score.

### Live Sensor Metric Cards

Five cards displayed in a row:
- Soil Moisture (%)
- Temperature (°C)
- Humidity (%)
- Light Intensity (lux)
- ML Prediction with confidence score (colour-coded)

### Historical Trend Charts

Four line charts in a 2x2 grid showing the last 50 sensor readings over time — one chart per sensor, with colour fill below the line for visual clarity.

### Health Status Pie Chart

Shows the distribution of predicted health classes in the current session — useful to see which conditions are being detected most frequently.

### ML Evaluation Plots

The four charts generated by `ml_model.py` are displayed directly in the dashboard:
- Confusion Matrix heatmap
- Feature Importance bar chart
- Sensor Value Distributions (histograms per class)
- 5-Fold Cross-Validation accuracy scores

### Dataset Preview Table

Shows the first 20 rows of `dataset.csv` so the user can inspect the raw training data.

---

## 12. Results and Discussion

### 12.1 Model Performance

The Random Forest classifier achieved **99.58% test accuracy** and **98.50% +/- 0.42% cross-validation accuracy**. This is excellent performance showing the model has successfully learned to distinguish between all four plant health conditions.

The cross-validation standard deviation of only 0.42% shows the model is **consistent and not overfitting** — it performs reliably across different subsets of the data.

### 12.2 Why Such High Accuracy?

The high accuracy results from careful dataset design. Each health class was assigned clearly distinct sensor value ranges:

- "Needs Water" is the only class with soil moisture around 20% — very easy to detect.
- "High Temp Stress" is the only class with temperature around 38°C — clearly higher than all others at 22–26°C.
- "Low Light" is the only class with light intensity around 200 lux — far below all others at 650–700 lux.

This reflects the real world accurately — genuine plant stress conditions produce sensor readings that are clearly different from healthy readings.

### 12.3 The One Misclassification

Out of 240 test samples, only 1 was misclassified: a "Needs Water" sample was predicted as "High Temperature Stress". This is understandable because both conditions involve moderately reduced humidity (45% vs 30%), and in this particular sample the values happened to fall in an overlapping region. This is acceptable and expected — even in real sensor data, edge cases occur.

### 12.4 Feature Importance Insights

Light intensity was the most important feature with an importance score of approximately 0.42. This makes sense: the "Low Light Condition" class reads around 200 lux while all other classes read around 650–700 lux — a 3.5x difference that is trivially easy for the model to detect.

Humidity was the least important feature (importance ~0.12) because its value ranges overlap the most across classes — it alone cannot uniquely identify any single class.

### 12.5 Dashboard Real-Time Performance

The live dashboard successfully:
- Generates and displays new sensor readings on every refresh
- Runs ML inference in under 1 millisecond (Random Forest is very fast)
- Maintains 50-reading history for trend visualisation
- Correctly triggers red alert banners for non-healthy conditions
- Allows manual testing with any custom sensor values via sliders

---

## 13. Challenges and How We Solved Them

| Challenge | Solution |
|-----------|----------|
| No physical hardware available | Used NumPy Gaussian distributions to simulate realistic sensor noise based on real-world sensor behaviour described in agriculture literature |
| Designing realistic sensor profiles | Studied IoT agriculture research papers to set appropriate mean and standard deviation values for each health condition |
| Keeping the dashboard truly live | Used Streamlit `session_state` to maintain reading history across re-runs and `st.rerun()` for auto-refresh every 3 seconds |
| Loading ML model efficiently | Used `@st.cache_resource` so the 200-tree Random Forest is loaded only once and reused across all page refreshes |
| Balanced dataset for fair training | Generated exactly 300 samples per class and used stratified train/test split to ensure equal class representation |
| Model persistence across sessions | Used Python's built-in `pickle` module to serialise the trained classifier to disk, avoiding retraining on every dashboard launch |

---

## 14. Conclusion

This project successfully demonstrates a complete, functional, end-to-end IoT + Machine Learning system for smart plant health monitoring — running entirely on a laptop without any physical hardware.

**Key achievements:**

1. **Realistic Simulation** — 1,200 sensor readings generated with Gaussian noise profiles that accurately represent real IoT sensor behaviour across 4 distinct plant health conditions.

2. **High-Accuracy ML Model** — Random Forest classifier achieving 99.58% test accuracy and 98.50% cross-validation accuracy, with a complete professional evaluation including confusion matrix, feature importance scores, and per-class precision/recall/F1 metrics.

3. **Live Dashboard** — A fully functional Streamlit web application that simulates real-time sensor data streaming, runs live ML predictions, displays historical sensor trends, and triggers colour-coded health alerts.

4. **Standard IoT Architecture** — The project follows the accepted 4-layer IoT architecture (Perception → Network → Processing → Application), making it directly extensible to a real hardware deployment by replacing only the sensor simulation layer.

5. **Reproducibility** — The entire pipeline is deterministic (random seed 42) and runs with just three commands, making it easy to verify, reproduce, and extend.

This project proves that the combination of IoT architecture principles and Machine Learning can produce a practical, intelligent plant monitoring solution that goes far beyond simple sensor dashboards — it interprets the data, classifies the plant's condition, and acts on it in real time.

---

## 15. Future Scope

| Enhancement | Description | Difficulty |
|-------------|-------------|------------|
| **Real Hardware Integration** | Replace `sensor_simulation.py` with an ESP32 reading real sensors via serial port or MQTT broker | Medium |
| **Cloud Storage** | Stream sensor readings to AWS IoT Core or Firebase Realtime Database for remote monitoring and historical storage | Medium |
| **Mobile App** | Build a Flutter or React Native application that connects to a FastAPI backend serving the ML model | High |
| **LSTM Time-Series ML** | Replace Random Forest with an LSTM neural network that captures temporal patterns (e.g., moisture dropping over several hours) | High |
| **Automated Irrigation** | Trigger a relay-controlled water pump when "Needs Water" is predicted for 3+ consecutive readings | Medium |
| **Multi-Plant Support** | Monitor multiple plants simultaneously, each with its own sensor cluster and individual dashboard panel | Medium |
| **Email and SMS Alerts** | Send automated notifications via Twilio SMS or Gmail SMTP when the plant needs urgent attention | Low |
| **Leaf Disease Detection** | Add a Raspberry Pi camera with a CNN model (MobileNetV2) for visual leaf disease classification from images | High |
| **Anomaly Detection** | Add an Isolation Forest to detect completely novel stress patterns not covered by the 4 trained classes | Medium |
| **Solar-Powered Nodes** | Design battery and solar-charged wireless sensor nodes for outdoor garden deployment using LoRa | High |

---

## 16. References

1. Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D. (2018). *Machine Learning in Agriculture: A Review.* **Sensors**, 18(8), 2674. https://doi.org/10.3390/s18082674

2. Talavera, J. M., Tobón, L. E., Gómez, J. A., Culman, M. A., Aranda, J. M., Parra, D. T., & Garreta, L. E. (2017). *Review of IoT applications in agro-industrial and environmental fields.* **Computers and Electronics in Agriculture**, 142, 283–297.

3. Breiman, L. (2001). *Random Forests.* **Machine Learning**, 45(1), 5–32.

4. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine Learning in Python.* **Journal of Machine Learning Research**, 12, 2825–2830.

5. Harris, C. R., et al. (2020). *Array programming with NumPy.* **Nature**, 585, 357–362.

6. Python Software Foundation. (2024). *Python Language Reference.* https://www.python.org

7. Streamlit Inc. (2024). *Streamlit Documentation.* https://docs.streamlit.io

8. Scikit-learn Developers. (2024). *Scikit-learn User Guide.* https://scikit-learn.org/stable/

---

*Academic Project — Simulation-based IoT and Machine Learning System*  
*Developed using Python, scikit-learn, and Streamlit*
