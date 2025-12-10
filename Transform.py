import pandas as pd
from Extraction import extract_blocks_range, get_latest_block_number

def records_to_dataframe(records):
    if not records:
        print("[WARN] Empty records in transform.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df[df["gas_limit"] > 0]
    df["gas_utilization"] = df["gas_used"] / df["gas_limit"]
    df = df[(df["gas_utilization"] >= 0) & (df["gas_utilization"] <= 1.5)]
    return df
    
def coll():
    match2 = []
    for i in range(1,101):
        url=f'https://op.gg/lol/leaderboards/tier?region=kr&page={i}'
        rq = requests.get(url)
        html = rq.text
        #html = '<a data-tooltip-id="opgg-tooltip" data-tooltip-content="Rengar" href="/lol/champions/rengar/build">'
        match = re.findall(r'<a[^>]*data-tooltip-content="(.*?)"[^>]*href="/lol/champions/[^"]+/build"', html)
        match2.extend(match)
        
def make_minute_level_features(df):
    if df.empty:
        print("[WARN] Empty DataFrame in make_minute_level_features.")
        return df

    df = df.copy()
    df["minute"] = df["timestamp_dt"].dt.floor("min")

    agg = (
        df.groupby("minute")
          .agg(
              avg_priority_fee_gwei=("avg_priority_fee_gwei", "mean"),
              avg_gas_utilization=("gas_utilization", "mean"),
              total_tx_count=("tx_count", "sum"),
              block_count=("block_number", "count"),
          )
          .reset_index()
          .sort_values("minute")
    )

    agg["roll_fee_10m"] = (
        agg["avg_priority_fee_gwei"].rolling(window=10, min_periods=1).mean()
    )
    agg["roll_fee_30m"] = (
        agg["avg_priority_fee_gwei"].rolling(window=30, min_periods=1).mean()
    )

    return agg

if __name__ == "__main__":
    latest = get_latest_block_number()
    start = max(0, latest - 200)
    records = extract_blocks_range(start, latest)

    df_blocks = records_to_dataframe(records)
    print(df_blocks.head())

    df_minute = make_minute_level_features(df_blocks)
    print(df_minute.head())
