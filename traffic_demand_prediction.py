import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# =====================================================
# LOAD DATA
# =====================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train Shape:", train.shape)
print("Test Shape :", test.shape)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_features(df):

    df = df.copy()

    # ---------------------------
    # Time Features
    # ---------------------------

    df["timestamp"] = df["timestamp"].astype(str)

    df["hour"] = (
        df["timestamp"]
        .str.split(":")
        .str[0]
        .astype(int)
    )

    df["minute"] = (
        df["timestamp"]
        .str.split(":")
        .str[1]
        .astype(int)
    )

    df["total_minutes"] = (
        df["hour"] * 60 +
        df["minute"]
    )

    # Cyclic Time Encoding

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # ---------------------------
    # Day Features
    # ---------------------------

    df["day_mod7"] = df["day"] % 7

    df["weekend_flag"] = (
        df["day_mod7"] >= 5
    ).astype(int)

    # ---------------------------
    # Geohash Features
    # ---------------------------

    df["geo3"] = (
        df["geohash"]
        .astype(str)
        .str[:3]
    )

    df["geo4"] = (
        df["geohash"]
        .astype(str)
        .str[:4]
    )

    df["geo5"] = (
        df["geohash"]
        .astype(str)
        .str[:5]
    )

    df["geo_len"] = (
        df["geohash"]
        .astype(str)
        .str.len()
    )

    # ---------------------------
    # Interaction Features
    # ---------------------------

    df["road_weather"] = (
        df["RoadType"]
        .fillna("missing")
        .astype(str)
        + "_"
        +
        df["Weather"]
        .fillna("missing")
        .astype(str)
    )

    df["road_vehicle"] = (
        df["RoadType"]
        .fillna("missing")
        .astype(str)
        + "_"
        +
        df["LargeVehicles"]
        .fillna("missing")
        .astype(str)
    )

    df["road_landmark"] = (
        df["RoadType"]
        .fillna("missing")
        .astype(str)
        + "_"
        +
        df["Landmarks"]
        .fillna("missing")
        .astype(str)
    )

    return df


train = create_features(train)
test = create_features(test)

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

cat_cols = [

    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",

    "geo3",
    "geo4",
    "geo5",

    "road_weather",
    "road_vehicle",
    "road_landmark"
]

for col in cat_cols:

    train[col] = (
        train[col]
        .fillna("missing")
        .astype(str)
    )

    test[col] = (
        test[col]
        .fillna("missing")
        .astype(str)
    )

train["Temperature"] = train["Temperature"].fillna(
    train["Temperature"].median()
)

test["Temperature"] = test["Temperature"].fillna(
    train["Temperature"].median()
)

# =====================================================
# FEATURE LIST
# =====================================================

TARGET = "demand"

DROP_COLS = [
    "Index",
    "timestamp",
    TARGET
]

FEATURES = [
    c
    for c in train.columns
    if c not in DROP_COLS
]

X = train[FEATURES]
y = train[TARGET]

X_test = test[FEATURES]

# =====================================================
# CATBOOST CAT FEATURES
# =====================================================

cat_features = [

    X.columns.get_loc(col)

    for col in cat_cols

    if col in X.columns
]

# =====================================================
# 5 FOLD CV
# =====================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

scores = []

# =====================================================
# TRAIN
# =====================================================

for fold, (tr_idx, val_idx) in enumerate(
    kf.split(X),
    start=1
):

    print(f"\n========== Fold {fold} ==========")

    X_train = X.iloc[tr_idx]
    y_train = y.iloc[tr_idx]

    X_valid = X.iloc[val_idx]
    y_valid = y.iloc[val_idx]

    model = CatBoostRegressor(

        iterations=5000,

        learning_rate=0.03,

        depth=10,

        loss_function="RMSE",

        eval_metric="R2",

        random_seed=42,

        verbose=200
    )

    model.fit(

        X_train,
        y_train,

        cat_features=cat_features,

        eval_set=(
            X_valid,
            y_valid
        ),

        use_best_model=True,

        early_stopping_rounds=300
    )

    preds = model.predict(X_valid)

    fold_r2 = r2_score(
        y_valid,
        preds
    )

    print(
        f"Fold {fold} R2 = {fold_r2:.6f}"
    )

    scores.append(fold_r2)

    oof[val_idx] = preds

    test_preds += (
        model.predict(X_test)
        / kf.n_splits
    )

# =====================================================
# RESULTS
# =====================================================

overall_r2 = r2_score(
    y,
    oof
)

print("\n" + "=" * 50)

print("Fold Scores")

print(scores)

print(
    f"\nMean Fold R2 : {np.mean(scores):.6f}"
)

print(
    f"Overall CV R2: {overall_r2:.6f}"
)

print("=" * 50)

# =====================================================
# SUBMISSION
# =====================================================

submission = pd.DataFrame({

    "Index": test["Index"],

    "demand": test_preds
})

submission.to_csv(
    "submission.csv",
    index=False
)

print(
    "\nsubmission.csv generated successfully"
)

print(submission.head())