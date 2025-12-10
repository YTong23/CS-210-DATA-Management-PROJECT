import os
import psycopg2
import psycopg2.extras
from Extraction import extract_blocks_range, get_latest_block_number
from Transform import records_to_dataframe, make_minute_level_features

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB", "ethgas")
PG_USER = os.getenv("PG_USER", "tongyang")
PG_PASSWORD = os.getenv("PG_PASSWORD", "123")

def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
    )

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS blocks (
            block_number BIGINT PRIMARY KEY,
            ts           TIMESTAMPTZ NOT NULL,
            gas_used     BIGINT,
            gas_limit    BIGINT,
            tx_count     INTEGER,
            avg_priority_fee_gwei DOUBLE PRECISION
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS minute_stats (
            minute_ts              TIMESTAMPTZ PRIMARY KEY,
            avg_priority_fee_gwei  DOUBLE PRECISION,
            avg_gas_utilization    DOUBLE PRECISION,
            total_tx_count         BIGINT,
            block_count            INTEGER,
            roll_fee_10m           DOUBLE PRECISION,
            roll_fee_30m           DOUBLE PRECISION
        );
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_blocks_ts ON blocks(ts);"
    )
    
    cur.execute(
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS minute_stats_mv AS
    SELECT
        date_trunc('minute', ts) AS minute_ts,
        avg(avg_priority_fee_gwei) AS avg_priority_fee_gwei,
        avg(gas_used::float / gas_limit) AS avg_gas_utilization,
        sum(tx_count) AS total_tx_count,
        count(*) AS block_count
    FROM blocks
    GROUP BY 1
    ORDER BY 1;
    """
)

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_msmv_ts ON minute_stats_mv(minute_ts);"
    )

    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] DB initialized.")

def load_blocks(records):
    if not records:
        print("[WARN] No records to load into blocks.")
        return

    conn = get_conn()
    cur = conn.cursor()

    rows = []
    for r in records:
        rows.append(
            (
                r["block_number"],
                r["timestamp"],
                r["gas_used"],
                r["gas_limit"],
                r["tx_count"],
                r["avg_priority_fee_gwei"],
            )
        )

    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO blocks (
            block_number, ts, gas_used, gas_limit, tx_count, avg_priority_fee_gwei
        )
        VALUES %s
        ON CONFLICT (block_number) DO NOTHING;
        """,
        rows,
        template="(%s, to_timestamp(%s), %s, %s, %s, %s)",
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"[INFO] Inserted {len(rows)} rows into blocks.")

def load_minute_stats(df_minute):
    if df_minute is None or df_minute.empty:
        print("[WARN] Empty df_minute, nothing to load.")
        return

    conn = get_conn()
    cur = conn.cursor()

    rows = []
    for _, row in df_minute.iterrows():
        rows.append(
            (
                row["minute"],
                row["avg_priority_fee_gwei"],
                row["avg_gas_utilization"],
                row["total_tx_count"],
                row["block_count"],
                row["roll_fee_10m"],
                row["roll_fee_30m"],
            )
        )

    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO minute_stats (
            minute_ts,
            avg_priority_fee_gwei,
            avg_gas_utilization,
            total_tx_count,
            block_count,
            roll_fee_10m,
            roll_fee_30m
        )
        VALUES %s
        ON CONFLICT (minute_ts) DO NOTHING;
        """,
        rows,
        template="(%s, %s, %s, %s, %s, %s, %s)",
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"[INFO] Inserted {len(rows)} rows into minute_stats.")

if __name__ == "__main__":
    init_db()

    latest = get_latest_block_number()
    start = max(0, latest - 300)

    records = extract_blocks_range(start, latest)

    df_blocks = records_to_dataframe(records)
    df_minute = make_minute_level_features(df_blocks)

    load_blocks(records)
    load_minute_stats(df_minute)
