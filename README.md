# 🚦 Traffic Demand Prediction — Gridlock Hackathon 2.0

![Score](https://img.shields.io/badge/R²%20Score-93.94%25-brightgreen)
![HackerEarth](https://img.shields.io/badge/HackerEarth-Accepted-blue)
![Flipkart](https://img.shields.io/badge/Organized%20by-Flipkart-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> 🏆 Achieved **93.94% R² Score** (Score: 90.63/100) on the 
> Gridlock Hackathon 2.0 — Traffic Demand Prediction challenge 
> organized by **Flipkart** on HackerEarth.

---

## 📌 Problem Statement

Cities worldwide face increasing traffic congestion, disrupting 
transportation and economic growth. This project builds an 
AI-powered system to predict **traffic demand** at specific 
locations and timestamps, providing insights into:

- Passenger travel patterns
- Booking behavior trends
- Trip cancellation predictions
- Urban mobility demand forecasting

---

## 📊 Dataset

| File | Rows | Columns |
|---|---|---|
| train.csv | 77,299 | 11 |
| test.csv | 41,778 | 10 |
| submission.csv | 41,778 | 2 |

### Features
| Column | Description |
|---|---|
| geohash | Geographic location encoding |
| day | Day of record |
| timestamp | Time of record |
| RoadType | Type of road |
| NumberofLanes | Number of lanes |
| LargeVehicles | Large vehicle permission |
| Landmarks | Nearby landmarks |
| Temperature | Location temperature |
| Weather | Weather condition |
| **demand** | **Target variable** |

---

## 🎯 Evaluation Metric

```python
score = max(0, 100 * (metrics.r2_score(actual, predicted)))
```

**Final Score: 90.63 / 100 (R² = 0.9394)**

---

## ✨ Solution Highlights

### 🔧 Feature Engineering (43+ Features)

**Geohash Hierarchical Extraction**
- Prefix lengths 3, 4, 5 for spatial zoom pyramid
- Approximate lat/lon decoding
- Coarse-to-fine location groupings

**Cyclic Time Encoding**
- sin/cos transformation for hour (prevents 23→0 discontinuity)
- Peak hour flags (morning peak, evening peak, night, midday)
- Day-of-week cyclic features

**Interaction Features**
- `geohash × hour` — location-time demand patterns
- `weather × hour` — weather impact by time of day
- `road_type × lanes` — road capacity interactions
- `temperature × weather` — environmental combinations

**Target Encoding**
- K-fold leak-free encoding (5 folds)
- Laplace smoothing (m=20) for rare categories
- Applied to 10 high-cardinality columns

---

### 🤖 Models Used

| Model | OOF R² | Notes |
|---|---|---|
| **LightGBM** | 0.9394 | Best single model |
| **CatBoost** | ~0.935 | Native categorical handling |
| **XGBoost** | ~0.930 | Diverse predictions |
| **ExtraTrees** | ~0.910 | Low variance, high diversity |
| **RandomForest** | ~0.905 | Stable bagged ensemble |
| **Weighted Ensemble** | **0.9394+** | R²-proportional weights |

---

### ⚡ Hyperparameter Optimization

Used **Optuna TPE Sampler** for intelligent search:
- LightGBM: 40 trials → Best R² 0.9394
- CatBoost: 30 trials
- Each trial evaluates full 5-fold CV (honest HPO)

Best LightGBM Parameters found:
```python
{
    "n_estimators": 1595,
    "learning_rate": 0.01021,
    "num_leaves": 169,
    "max_depth": 10,
    "min_child_samples": 46,
    "subsample": 0.9113
}
```

---

## 🏗️ Pipeline Architecture
