# Polymarket NVIDIA CEO forecast entry

Entry for the make.fun competition `polymarket-nvidia-ceo-forecast-01`.

The program reads one case file `/input/<id>.in` (canonical JSON with the 72 hours of
one-minute Yes prices of each trading market of Polymarket's "Next CEO of NVIDIA?"
event up to a checkpoint) and writes `/output/<id>.out`: five quantiles
(5%, 25%, 50%, 75%, 95%) of each market's price at each of the 240 minutes after the
checkpoint.

## Model

* The checkpoint price `p0` is each market's last observed price.
* Empirical quantiles of h-step price changes are estimated for h in
  {1, 5, 15, 60, 240} from the full available history, with exponential age weights
  (tau = 240 minutes) so recent behaviour dominates. Each curve is centered on its
  median.
* That empirical curve is blended 60/40 with a random-walk shape built from robust
  MAD volatility of recent one-minute changes (with a floor), and interpolated in
  log-h. Young markets with no direct h-step sample extrapolate the nearest curve
  like a random walk.
* A `0.001 * sqrt(h)` 5-95 spread floor prevents collapse; a small
  `-0.06 * p0 * h/240` decay drift corrects the observed upward bias of checkpoint
  prices.
* Every value is clipped to `[0, 1]`, rounded to 6 decimals and made non-decreasing.

The program is deterministic, uses only the Python 3.13 standard library, opens no
network connections, and writes the kit's exact output format.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
