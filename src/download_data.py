"""Download the fixed Yahoo Finance price snapshot used in this project."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
STOCKS = {
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "TCS.NS": "Information technology", "INFY.NS": "Information technology",
    "HCLTECH.NS": "Information technology", "ITC.NS": "Consumer goods",
    "HINDUNILVR.NS": "Consumer goods", "RELIANCE.NS": "Energy and diversified",
    "LT.NS": "Engineering and construction",
}
START, END = "2023-01-01", "2026-01-01"  # end is exclusive

def main():
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    series = []
    for ticker in [*STOCKS, "^NSEI"]:
        frame = yf.download(ticker, start=START, end=END, auto_adjust=False,
                            progress=False, threads=False)
        if frame.empty:
            raise RuntimeError(f"No data for {ticker}; no simulated data will be substituted.")
        s = frame["Adj Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        series.append(s.rename(ticker))
        print(f"Downloaded {ticker}: {len(s)} prices", flush=True)
    prices = pd.concat(series, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    target = out / "adjusted_prices.csv"
    prices.to_csv(target, float_format="%.10f")
    metadata = {
        "provider": "Yahoo Finance", "retrieval_tool": "yfinance",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": START, "requested_end_exclusive": END,
        "price_field": "Adj Close", "stocks": STOCKS, "benchmark": "^NSEI",
        "rows": len(prices), "missing_by_column": prices.isna().sum().to_dict(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_urls": {t: "https://finance.yahoo.com/quote/" + t + "/history/" for t in [*STOCKS, "^NSEI"]},
        "note": "Fixed educational snapshot; vendor adjustments may change on a later download. NIFTY 50 is a price index, not a total-return benchmark."
    }
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
