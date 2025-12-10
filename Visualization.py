import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

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

def load_minute_stats(limit_hours=6):
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
        WHERE minute_ts >= NOW() - INTERVAL %s
        ORDER BY minute_ts;
    """
    df = pd.read_sql(query, conn, params=(f"{limit_hours} hours",))
    conn.close()
    if df.empty:
        print("[WARN] No data loaded from minute_stats.")
    return df

def plot_fee_timeseries(df):
    plt.figure()
    plt.plot(df["minute_ts"], df["avg_priority_fee_gwei"])
    plt.xlabel("Time")
    plt.ylabel("Avg Priority Fee (Gwei)")
    plt.title("Per-minute Avg Priority Fee (Gwei)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_fee_with_rolling(df):
    plt.figure()
    plt.plot(df["minute_ts"], df["avg_priority_fee_gwei"], label="Per-minute avg fee")
    plt.plot(df["minute_ts"], df["roll_fee_10m"], label="10-min rolling avg")
    plt.plot(df["minute_ts"], df["roll_fee_30m"], label="30-min rolling avg")
    plt.xlabel("Time")
    plt.ylabel("Fee (Gwei)")
    plt.title("Gas Fee with 10-min and 30-min Rolling Averages")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_utilization_timeseries(df):
    plt.figure()
    plt.plot(df["minute_ts"], df["avg_gas_utilization"])
    plt.xlabel("Time")
    plt.ylabel("Avg Gas Utilization")
    plt.title("Per-minute Avg Gas Utilization")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_util_vs_fee_scatter(df):
    plt.figure()
    plt.scatter(df["avg_gas_utilization"], df["avg_priority_fee_gwei"])
    plt.xlabel("Avg Gas Utilization")
    plt.ylabel("Avg Priority Fee (Gwei)")
    plt.title("Gas Utilization vs Priority Fee")
    plt.tight_layout()
    plt.show()

def plot_fee_histogram(df):
    plt.figure()
    plt.hist(df["avg_priority_fee_gwei"].dropna(), bins=30)
    plt.xlabel("Avg Priority Fee (Gwei)")
    plt.ylabel("Count")
    plt.title("Distribution of Per-minute Avg Priority Fee")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    if not os.path.exists("plots"):
        os.makedirs("plots")

    df = load_minute_stats(limit_hours=6)

    if not df.empty:
        plt.switch_backend("Agg")

        plot_fee_timeseries(df)
        plt.savefig("plots/fee_timeseries.png")
        plt.close()

        plot_fee_with_rolling(df)
        plt.savefig("plots/fee_rolling.png")
        plt.close()

        plot_utilization_timeseries(df)
        plt.savefig("plots/utilization_timeseries.png")
        plt.close()

        plot_util_vs_fee_scatter(df)
        plt.savefig("plots/util_vs_fee.png")
        plt.close()

        plot_fee_histogram(df)
        plt.savefig("plots/fee_hist.png")
        plt.close()

    print("Plots saved in ./plots/")