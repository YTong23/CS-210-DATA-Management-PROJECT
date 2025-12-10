import os
import psycopg2
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import numpy as np
from xgboost import XGBRegressor
import time
import psycopg2
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB", "ethgas")
PG_USER = os.getenv("PG_USER", "tongyang")
PG_PASSWORD = os.getenv("PG_PASSWORD", "123")

def test_query_latency():
    conn = get_conn()
    cur = conn.cursor()
    t0 = time.time()
    cur.execute("SELECT * FROM minute_stats_mv WHERE minute_ts >= NOW() - INTERVAL '1 hour';")
    cur.fetchall()
    t1 = time.time()
    print(f"[Latency] Query time = {1000*(t1-t0):.2f} ms")
    cur.close()
    conn.close()

def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
    )

def load_all_minute_stats():
    conn = get_conn()
    query = """
        SELECT
            minute_ts,
            avg_priority_fee_gwei,
            avg_gas_utilization,
            total_tx_count,
            block_count,
            roll_fee_10m,
            roll_fee_30m
        FROM minute_stats
        ORDER BY minute_ts;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df = df.dropna(subset=["avg_priority_fee_gwei"]).reset_index(drop=True)
    return df

def build_supervised_dataset(df, horizon=10):
    df = df.copy()
    df["target_10m_ahead"] = df["avg_priority_fee_gwei"].shift(-horizon)
    df["fee_lag_1"] = df["avg_priority_fee_gwei"].shift(1)
    df["fee_lag_2"] = df["avg_priority_fee_gwei"].shift(2)
    df["fee_lag_3"] = df["avg_priority_fee_gwei"].shift(3)

    feature_cols = [
        "avg_priority_fee_gwei",
        "roll_fee_10m",
        "roll_fee_30m",
        "avg_gas_utilization",
        "total_tx_count",
        "fee_lag_1",
        "fee_lag_2",
        "fee_lag_3",
    ]

    df = df.dropna(subset=feature_cols + ["target_10m_ahead"]).reset_index(drop=True)
    X = df[feature_cols].values
    y = df["target_10m_ahead"].values
    ts = df["minute_ts"].values
    return X, y, ts, df

def train_test_split_time_series(X, y, ts, test_ratio=0.2):
    n = len(X)
    test_size = int(n * test_ratio)
    train_size = n - test_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    ts_train = ts[:train_size]
    ts_test = ts[train_size:]

    return X_train, y_train, X_test, y_test, ts_train, ts_test

def eval_regression(y_true, y_pred, name="model"):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{name}] RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    return rmse, mae

def directional_accuracy(y_true, y_pred):
    st = np.sign(np.diff(y_true))
    sp = np.sign(np.diff(y_pred))
    return (st == sp).mean()


def run_baselines(df, horizon=10):
    df = df.copy()
    df["target_10m_ahead"] = df["avg_priority_fee_gwei"].shift(-horizon)
    df = df.dropna(subset=["target_10m_ahead", "roll_fee_10m"])

    n = len(df)
    test_size = int(n * 0.2)
    train_size = n - test_size

    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    y_test = df_test["target_10m_ahead"].values

    y_pred_curr = df_test["avg_priority_fee_gwei"].values
    eval_regression(y_test, y_pred_curr, name="Baseline_current_fee")

    y_pred_roll10 = df_test["roll_fee_10m"].values
    eval_regression(y_test, y_pred_roll10, name="Baseline_roll10")

def run_tree_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval_regression(y_test, y_pred, name="RandomForest")
    return model, y_pred

def run_xgboost_model(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval_regression(y_test, y_pred, name="XGBoost")
    da = directional_accuracy(y_test, y_pred)
    print(f"[XGBoost] Directional Accuracy = {da:.4f}")
    return model, y_pred


if __name__ == "__main__":
    df = load_all_minute_stats()
    print(f"[INFO] Loaded {len(df)} rows from minute_stats.")

    run_baselines(df, horizon=10)

    X, y, ts, df_supervised = build_supervised_dataset(df, horizon=10)
    print(f"[INFO] Supervised dataset shape: X={X.shape}, y={y.shape}")

    X_train, y_train, X_test, y_test, ts_train, ts_test = train_test_split_time_series(
        X, y, ts, test_ratio=0.2
    )
    print(f"[INFO] Train size={len(y_train)}, Test size={len(y_test)}")

    model, y_pred = run_tree_model(X_train, y_train, X_test, y_test)

    da = directional_accuracy(y_test, y_pred)
    print(f"[model] Directional Accuracy = {da:.4f}")
    
    model_xgb, y_pred_xgb = run_xgboost_model(X_train, y_train, X_test, y_test)

    test_query_latency()