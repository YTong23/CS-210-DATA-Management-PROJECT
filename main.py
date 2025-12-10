import time
from Extraction import *
from Transform  import *
from Loading import *
from Model import *
def refresh_materialized_view():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("REFRESH MATERIALIZED VIEW minute_stats_mv;")
    conn.commit()
    cur.close()
    conn.close()

def main():
    init_db()

    latest = get_latest_block_number()
    start = latest - 300
    records = extract_blocks_range(start, latest)

    df_blocks = records_to_dataframe(records)
    df_minute = make_minute_level_features(df_blocks)

    load_blocks(records)
    load_minute_stats(df_minute)

    refresh_materialized_view()

    df = load_all_minute_stats()
    run_baselines(df, horizon=10)

    X, y, ts, df_supervised = build_supervised_dataset(df, horizon=10)
    X_train, y_train, X_test, y_test, ts_train, ts_test = train_test_split_time_series(
        X, y, ts, test_ratio=0.2
    )

    model_rf, pred_rf = run_tree_model(X_train, y_train, X_test, y_test)
    da_rf = directional_accuracy(y_test, pred_rf)
    print(f"[RF] Directional Accuracy = {da_rf:.4f}")

    model_xgb, pred_xgb = run_xgboost_model(X_train, y_train, X_test, y_test)
    da_xgb = directional_accuracy(y_test, pred_xgb)
    print(f"[XGB] Directional Accuracy = {da_xgb:.4f}")

    test_query_latency()

if __name__ == "__main__":
    main()

