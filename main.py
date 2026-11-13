#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, five quantiles a minute.

Reads a makefun/polymarket-case/v1 file from the input directory and writes the
kit's forecast format to the output directory.  Standard library only, offline,
deterministic.

Model (per market, checkpoint price p0):
  * empirical quantiles of h-step price changes for h in {1,5,15,60,240}, taken
    from the whole available history but exponentially age-weighted (tau = 240
    minutes), and centered on zero;
  * blended 60/40 with a random-walk shape estimated from recent one-minute
    changes (robust MAD volatility, floored);
  * quantiles interpolated in log-h, with a random-walk extrapolation when a
    young market has no direct h-step sample;
  * a 0.001*sqrt(h) 5-95 spread floor;
  * a small price-proportional decay drift, -0.06 * p0 * h/240;
  * everything clipped to [0,1], rounded to 6 decimals and made non-decreasing.
"""

from __future__ import annotations

import json
import math
import os
import sys

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
KNOTS = (1, 5, 15, 60, 240)
TAU = 240.0                 # age-weight decay, in minutes
BLEND = 0.6                 # weight of the empirical component
RW_FLOOR = 0.0003           # minimum one-minute robust volatility
SPREAD_FLOOR = 0.001        # 5-95 half-width per sqrt(minute)
DRIFT = -0.06               # price-proportional decay over 240 minutes
RECENT_MINUTES = 720
MAX_STD = 8.0


def weighted_quantiles(pairs: list[tuple[float, float]], qs: tuple[float, ...]) -> list[float]:
    """Quantiles of a weighted sample, by linear interpolation of the weighted CDF."""
    pairs.sort(key=lambda item: item[0])
    total = sum(w for _, w in pairs)
    out: list[float] = []
    n = len(pairs)
    cum = 0.0
    i = 0
    for q in qs:
        target = q * total
        while i < n - 1 and cum + pairs[i][1] < target:
            cum += pairs[i][1]
            i += 1
        value, weight = pairs[i]
        if i == 0 or weight <= 0.0:
            out.append(value)
        else:
            frac = (target - cum) / weight
            frac = 0.0 if frac < 0.0 else 1.0 if frac > 1.0 else frac
            out.append(pairs[i - 1][0] + (value - pairs[i - 1][0]) * frac)
    return out


def quantile_sorted(vals: list[float], q: float) -> float:
    n = len(vals)
    if n == 1:
        return vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    if lo >= n - 1:
        return vals[-1]
    return vals[lo] + (vals[lo + 1] - vals[lo]) * (pos - lo)


def median(vals: list[float]) -> float:
    return quantile_sorted(sorted(vals), 0.5) if vals else 0.0


def clamp_row(row: list[float]) -> list[float]:
    out: list[float] = []
    prev = 0.0
    for x in row:
        v = min(1.0, max(0.0, x))
        v = round(v, 6)
        if v == 0:
            v = 0.0
        if v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def rw_shape(changes: list[float]) -> list[float] | None:
    """Standardized one-minute change quantiles over all available history."""
    if len(changes) < 30:
        return None
    m = median(changes)
    sigma = 1.4826 * median([abs(x - m) for x in changes])
    nonzero = sum(1 for x in changes[-RECENT_MINUTES:] if x != 0.0)
    if sigma < 1e-6 or nonzero < 16:
        return None
    std = sorted(max(-MAX_STD, min(MAX_STD, (x - m) / sigma)) for x in changes)
    return [quantile_sorted(std, q) for q in QUANTILES]


def pooled_shape(shapes: list[list[float] | None]) -> list[float]:
    good = [s for s in shapes if s is not None]
    if not good:
        return [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
    return [sum(s[j] for s in good) / len(good) for j in range(len(QUANTILES))]


def empirical_curve(hist: list[float]) -> dict[int, list[float]]:
    """Age-weighted empirical h-step quantiles, centered on their median."""
    n = len(hist)
    table: dict[int, list[float]] = {}
    for h in KNOTS:
        if n <= h:
            continue
        pairs: list[tuple[float, float]] = []
        last = n - 1
        for i in range(0, n - h):
            age = last - (i + h)
            pairs.append((hist[i + h] - hist[i], math.exp(-age / TAU)))
        qs = weighted_quantiles(pairs, QUANTILES)
        center = qs[2]
        table[h] = [v - center for v in qs]
    return table


def forecast_market(hist: list[float], horizon: int, pool: list[float]) -> list[list[float]]:
    p0 = hist[-1]
    changes = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
    shape = rw_shape(changes)
    sh = shape if shape is not None else pool
    recent = changes[-RECENT_MINUTES:] if changes else []
    if recent:
        m = median(recent)
        sigma = 1.4826 * median([abs(x - m) for x in recent])
    else:
        sigma = 0.0
    sig = max(sigma, RW_FLOOR)

    table = empirical_curve(hist)
    available = sorted(table)
    rows: list[list[float]] = []

    for h in range(1, horizon + 1):
        if h <= KNOTS[0]:
            a = b = KNOTS[0]
            frac = 0.0
        elif h >= KNOTS[-1]:
            a = b = KNOTS[-1]
            frac = 0.0
        else:
            a, b, frac = KNOTS[0], KNOTS[1], 0.0
            for k in range(len(KNOTS) - 1):
                if KNOTS[k] <= h <= KNOTS[k + 1]:
                    a, b = KNOTS[k], KNOTS[k + 1]
                    frac = (math.log(h) - math.log(a)) / (math.log(b) - math.log(a))
                    break

        qa = table.get(a)
        qb = table.get(b)
        if qa is None and qb is None:
            if available:
                h0 = min(available, key=lambda k: abs(math.log(k) - math.log(h)))
                eq = [v * math.sqrt(h / h0) for v in table[h0]]
            else:
                eq = [0.0] * 5
        elif qa is None:
            eq = [v * math.sqrt(h / b) for v in qb]
        elif qb is None:
            eq = [v * math.sqrt(h / a) for v in qa]
        else:
            eq = [qa[j] + (qb[j] - qa[j]) * frac for j in range(5)]

        rw = [z * sig * math.sqrt(h) for z in sh]
        row = [p0 + BLEND * eq[j] + (1.0 - BLEND) * rw[j] for j in range(5)]

        floor = SPREAD_FLOOR * math.sqrt(h)
        row[0] = min(row[0], p0 - floor)
        row[4] = max(row[4], p0 + floor)

        shift = DRIFT * p0 * (h / 240.0)
        rows.append(clamp_row([v + shift for v in row]))
    return rows


def main() -> int:
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)

        if list(case.get("quantiles", QUANTILES)) != list(QUANTILES):
            raise ValueError(f"{name}: unexpected quantile levels")
        horizon = int(case["horizon_minutes"])
        markets = case["markets"]

        shapes = []
        for m in markets:
            prices = m.get("prices") or []
            changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
            shapes.append(rw_shape(changes) if len(changes) >= 30 else None)
        pool = pooled_shape(shapes)

        forecasts = []
        for m in markets:
            prices = m.get("prices") or []
            if not prices:
                raise ValueError(f"{name}: market {m.get('id')} has no prices")
            forecasts.append({"id": m["id"], "q": forecast_market(prices, horizon, pool)})

        path = os.path.join(out_dir, f"{case_id}.out")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"forecasts": forecasts}, fh, separators=(",", ":"), allow_nan=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
