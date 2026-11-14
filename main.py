#!/usr/bin/env python3
"""Produce deterministic BTC price forecasts for make.fun case files."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


HORIZON_MINUTES = 240
OLS_WINDOW = 600
EMA_SPAN = 480
DRIFT_LIMIT = 300.0


def parse_case(path: Path) -> tuple[str, list[float]]:
    """Load and validate one case, returning its id and close prices."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: tidak dapat membaca JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: root JSON harus object")
    if data.get("schema") != "makefun/btc-cent-case/v1":
        raise ValueError(f"{path.name}: schema tidak didukung")
    if data.get("competition") != "btc-cent-forecast-01":
        raise ValueError(f"{path.name}: competition tidak sesuai")
    case_id = data.get("case")
    if (not isinstance(case_id, str) or not case_id
            or Path(case_id).name != case_id or case_id != path.stem):
        raise ValueError(f"{path.name}: nilai case tidak valid")
    markets = data.get("markets")
    if not isinstance(markets, list) or not markets or not isinstance(markets[0], dict):
        raise ValueError(f"{path.name}: markets harus berisi market")
    prices = markets[0].get("prices")
    if not isinstance(prices, list) or not prices:
        raise ValueError(f"{path.name}: prices harus berupa array tidak kosong")
    try:
        values = [float(value) for value in prices]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path.name}: prices harus berupa angka") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path.name}: prices harus finite")
    if len(values) < OLS_WINDOW:
        raise ValueError(f"{path.name}: perlu sedikitnya {OLS_WINDOW} harga untuk OLS")
    return case_id, values


def ols_slope(prices: list[float]) -> float:
    """Return least-squares slope for the final 600 samples, in dollars/minute."""
    window = prices[-OLS_WINDOW:]
    count = len(window)
    mean_x = (count - 1) / 2.0
    mean_y = math.fsum(window) / count
    numerator = math.fsum((index - mean_x) * (value - mean_y)
                          for index, value in enumerate(window))
    denominator = count * (count * count - 1) / 12.0
    return numerator / denominator


def ema(prices: list[float]) -> float:
    """Calculate EMA over all prices, seeded with the first close."""
    alpha = 2.0 / (EMA_SPAN + 1)
    result = prices[0]
    for value in prices[1:]:
        result += alpha * (value - result)
    return result


def forecast(prices: list[float]) -> float:
    last = prices[-1]
    drift = max(-DRIFT_LIMIT, min(DRIFT_LIMIT, ols_slope(prices) * HORIZON_MINUTES))
    pull = 0.30 * (ema(prices) - last)
    # Clamp after rounding to keep the serialized value strictly inside bounds.
    raw_prediction = last + drift + pull
    if not math.isfinite(raw_prediction):
        raise ValueError("hasil forecast tidak finite")
    predicted = round(raw_prediction, 5)
    return max(0.00001, min(9_999_999.99999, predicted))


def main() -> int:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    if not input_dir.is_dir():
        raise ValueError(f"direktori input tidak ditemukan: {input_dir}")
    paths = sorted(input_dir.glob("*.in"), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"tidak ada file *.in di {input_dir}")

    # Validate and calculate everything before writing, so malformed input never
    # produces a partial set of apparently valid forecasts.
    results: list[tuple[str, float]] = []
    seen_ids: set[str] = set()
    for path in paths:
        case_id, prices = parse_case(path)
        if case_id in seen_ids:
            raise ValueError(f"case duplikat: {case_id}")
        seen_ids.add(case_id)
        results.append((case_id, forecast(prices)))

    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id, price in results:
        payload = {"forecasts": [{"case": case_id, "price": price}]}
        destination = output_dir / f"{case_id}.out"
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}")
