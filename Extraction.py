import os
import time
import requests
from typing import Dict, Any, List

ETHERSCAN_API_KEY = "2BN4UJFA2KI6XWQIKYHKFQVMSYI563U2RU"
BASE_URL = "https://api.etherscan.io/v2/api"

def call_etherscan(params: Dict[str, Any]):
    full_params = {"chainid": "1",**params,"apikey": ETHERSCAN_API_KEY,}

    for attempt in range(3):
        try:
            resp = requests.get(BASE_URL, params=full_params, timeout=10)
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                print("[WARN] Non-JSON response")
                time.sleep(0.5)
                continue

            data = resp.json()
            if data.get("status") == "0" and data.get("message") != "OK":
                print("[ERROR] Etherscan error:", data.get("message"), data.get("result"))
            return data

        except requests.RequestException as e:
            print(f"[WARN] Request failed (attempt {attempt+1}/3): {e}")
            time.sleep(0.5)

    raise RuntimeError("Failed after 3 attempts.")

def get_latest_block_number():
    params = {
        "module": "proxy",
        "action": "eth_blockNumber",
    }
    data = call_etherscan(params)
    hex_block = data["result"]
    return int(hex_block, 16)

def fetch_block_by_number(block_number: int):
    if block_number < 0:
        raise ValueError(f"Block number must be non-negative, got {block_number}")

    hex_block = hex(block_number)
    params = {
        "module": "proxy",
        "action": "eth_getBlockByNumber",
        "tag": hex_block,
        "boolean": "true",
    }
    data = call_etherscan(params)
    result = data.get("result")
    if result is None:
        raise RuntimeError(f"Missing 'result' for block {block_number}")
    return result

def parse_block(block_json: Dict[str, Any]):
    try:
        block_number = int(block_json["number"], 16)
        timestamp = int(block_json["timestamp"], 16)
        gas_used = int(block_json["gasUsed"], 16)
        gas_limit = int(block_json["gasLimit"], 16)
        txs = block_json.get("transactions", []) or []
        tx_count = len(txs)

        priority_fees_wei: List[int] = []
        for tx in txs:
            p = tx.get("maxPriorityFeePerGas")
            if p is not None:
                priority_fees_wei.append(int(p, 16))
        if priority_fees_wei:
            avg_priority_fee_wei = sum(priority_fees_wei) / len(priority_fees_wei)
            avg_priority_fee_gwei = avg_priority_fee_wei / 1e9
        else:
            avg_priority_fee_gwei = None

        return {
            "block_number": block_number,
            "timestamp": timestamp,
            "gas_used": gas_used,
            "gas_limit": gas_limit,
            "tx_count": tx_count,
            "avg_priority_fee_gwei": avg_priority_fee_gwei,
        }

    except KeyError as e:
        print(f"[WARN] Missing key : {e}")
        return None
    except ValueError as e:
        print(f"[WARN] Failed to parse: {e}")
        return None

def extract_blocks_range(start_block: int, end_block: int, sleep_seconds: float = 0.2):
    if start_block > end_block:
        raise ValueError(f"start_block must be <= end_block, got {start_block} > {end_block}")

    records: List[Dict[str, Any]] = []
    print(f"[INFO] Extracting blocks from {start_block} to {end_block} ...")

    for block_number in range(start_block, end_block + 1):
        print(f"[INFO] Fetching block {block_number} ...")
        raw_block = fetch_block_by_number(block_number)
        parsed = parse_block(raw_block)
        if parsed is not None:
            records.append(parsed)
        else:
            print(f"[WARN] Skipped block {block_number} due to parse error.")
        time.sleep(sleep_seconds)

    print(f"[INFO] Extraction finished. Got {len(records)} valid blocks.")
    return records

if __name__ == "__main__":
    latest = get_latest_block_number()
    start = max(0, latest - 19)
    data = extract_blocks_range(start, latest)
    for r in data[:3]:
        print(r)
