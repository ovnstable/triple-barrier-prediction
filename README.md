# Triple-barrier prediction — task overview

## Task 1. LP triple barrier

Predict the **`target_barrier`** column from the `target/` dataset and evaluate yourself on the **F1 score**.

- `target_barrier` is the outcome of taking a specific Uniswap liquidity provision `action` at a specific Arbitrum `blockNumber`.
  - Source: `data/target/{date}.parquet`
  - Primary key: (`blockNumber`, `action`)
  - Label: `target_barrier`

You may use any combination of the data shipped here (`steps/`, `binance/`, `hyperliquid/`, `uniswap/`) **or external sources of your own**, as long as you respect causality (no information after the step’s `datetime_ms`).

### Goal

For every (`blockNumber`, `action`) row, produce a prediction of `target_barrier` such that:

1. **F1 score is as high as possible** on held-out validation data.
2. **Performance is stable over time** — F1 should not degrade meaningfully across days / weeks of the validation window. Models that overfit one regime are penalized in practice; prefer features and training schemes that generalize.
3. **Inference on steps** - when ready, inference triple_barrier on all **blockNumber** provided in steps

### Evaluation

- Metric: **Macro F1 score** on `target_barrier`. Additionally, validate using the confusion matrix to avoid any abnormal predictions due to class imbalance.
- Always report **per-day F1** in addition to overall, to make the time-stability requirement visible.

## Task 2. Hedge triple barrier

Predict the following target columns from the `hedge_target/` dataset and evaluate yourself on the **F1 score**:

- `triple_barrier_sell_500ms`
- `triple_barrier_buy_500ms`
- `triple_barrier_sell_2000ms`
- `triple_barrier_buy_2000ms`

- Each target is the outcome of taking a specific hedge-side action at a specific timestamp.
  - Source: `data/hedge_target/{date}.parquet`
  - Join key: timestamp-aligned rows (as provided in the dataset)
  - Labels: the four `triple_barrier_*` columns above

You may use any combination of the data shipped here (`steps/`, `binance/`, `hyperliquid/`, `uniswap/`) **or external sources of your own**, as long as you respect causality (no information after the step’s `datetime_ms`).

### Goal

For every row, produce predictions for the four `triple_barrier_*` targets such that:

1. **F1 score is as high as possible** on held-out validation data.
2. **Performance is stable over time** — F1 should not degrade meaningfully across days / weeks of the validation window. Models that overfit one regime are penalized in practice; prefer features and training schemes that generalize.
3. **Inference on BBO timestamps** - when ready, run inference for all BBO timestamps in the evaluation horizon.

### Evaluation

- Metric: **Macro F1 score** on each target (`triple_barrier_sell_500ms`, `triple_barrier_buy_500ms`, `triple_barrier_sell_2000ms`, `triple_barrier_buy_2000ms`). Additionally, validate using confusion matrices to avoid abnormal predictions due to class imbalance.
- Always report **per-day F1** in addition to overall, to make the time-stability requirement visible.


---

## Unpacking zips (`data.zip.part-*`)

Large files in this repo are stored with **Git LFS**. Install the extension, then clone as usual; LFS objects download on `git pull` / `git checkout`.


```bash
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install

git lfs pull

unzip data.zip
```

---

## Data folder layout

Root: `data/` (e.g. `/data/triple-barrier-prediction/data`).

```
data/
├── steps/                 # one parquet per calendar day
├── target/                # primary labels / targets per day
├── hedge_target/          # alternative hedge-focused targets per day
├── binance/
│   ├── bbo/spot/          # best bid/offer snapshots per day
│   ├── orderbook/spot/    # L2 order book snapshots per day
│   └── trades/perp/       # perp trades per day
├── hyperliquid/
│   ├── orderbook/perp/
│   └── trades/perp/
└── uniswap/
    ├── liquidity/         # pool liquidity snapshots per day
    └── swaps/             # swap events per day
```
| Path | What it is | Join hint |
|------|------------|------------|
| **`steps/`** | Per-block backbone; includes **`datetime_ms`**. | Left side of joins; `blockNumber` + causal bound `datetime_ms`. |
| **`target/`** | LP triple-barrier labels / targets. | `blockNumber` on same-day file; respect `datetime_ms` if target rows carry timestamps. |
| **`hedge_target/`** | Hedge-focused target set. | Backward join on datetime. |
| **`binance/bbo/spot/`** | Spot BBO. | Same-day file; last snapshot with time ≤ `datetime_ms` per block. |
| **`binance/orderbook/spot/`** | Spot L2 books. | Slower compared to BBO, contains deeper levels. |
| **`binance/trades/perp/`** | Binance perp trades. | Trades with time ≤ `datetime_ms`; aggregate per block if needed. |
| **`hyperliquid/orderbook/perp/`** | HL perp book. | Same as Binance order book. |
| **`hyperliquid/trades/perp/`** | HL perp trades. | Same as Binance trades. |
| **`uniswap/swaps/`** | DEX swap events. | `blockNumber` (+ tx index if needed); gate on `datetime_ms` for strict recv-time sims. |
| **`uniswap/liquidity/`** | Pool liquidity snapshots. | `blockNumber` or as-of time ≤ `datetime_ms`. |

---

## Joining blockchain data to avoid look-ahead bias
- **Primary keys**
  - `steps/`: `blockNumber` only — used for **inference**. Carries `datetime_ms` (when the row was known).
  - `target/`: (`blockNumber`, `action`) — used for **training / validation**.

> `datetime_ms` lives on `steps`. To get it onto a `target`-keyed view, **join `target` onto `steps`** first. After that, attach off-chain feeds with `merge_asof` backward on `datetime_ms`, and on-chain feeds by `blockNumber`.


### Example

```python
import pandas as pd

day = "2026-04-02"

# 1) load core tables
steps  = pd.read_parquet(f"data/steps/{day}.parquet")     # PK: blockNumber, has datetime_ms
target = pd.read_parquet(f"data/target/{day}.parquet")    # PK: (blockNumber, action)

# 2) labelled view: bring datetime_ms onto target rows
labelled = target.merge(steps, on="blockNumber", how="left", suffixes=("", "_step"))

# 3a) off-chain join example: Binance spot BBO via as-of backward on datetime_ms
binance_bbo = pd.read_parquet(f"data/binance/bbo/spot/{day}.parquet") \
                .sort_values("datetime_ms")
labelled = labelled.sort_values("datetime_ms")

labelled = pd.merge_asof(
    labelled,
    binance_bbo,
    on="datetime_ms",
    direction="backward",  # last BBO at or before the step's datetime_ms
    suffixes=("", "_bbo"),
)

# 3b) on-chain join example: Uniswap swaps by block
uni_swaps = pd.read_parquet(f"data/uniswap/swaps/{day}.parquet")
labelled = labelled.merge(uni_swaps, on="blockNumber", how="left", suffixes=("", "_swap"))

# 4) inference view (no target): expand steps across actions
actions_df = pd.DataFrame({"action": [1, 2, 3, 4, 5]})  # use your real action set
infer = steps.merge(actions_df, how="cross")               # then attach feeds the same way
```
---

