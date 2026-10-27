# ZN Roll Liquidity: Predicting the Crossover

**The shift in trading between quarterly 10-Year U.S. Treasury Note futures contracts is highly predictable in this sample.** Across 27 ZN rolls from 2020–2026, a volume-based ridge model identifies the exact crossover observation in **14 of 19 held-out rolls**, with the remaining five just one observation away. Open interest usually migrates earlier, and daily execution proxies often favor dates before volume crossover.

Rolling replaces an expiring contract with the next quarterly contract. We study ZN alone to compare how positions, trading volume, and calendar-spread activity migrate, then ask whether predicting that migration helps identify favorable execution conditions.

The project asks two questions:

1. **Can we predict when trading moves into the next futures contract?** Specifically, when will the next contract first account for more than half of the combined trading volume in the expiring and next contracts? For example, if December trades 40,000 contracts and March trades 60,000, March has a 60% share and has crossed the threshold. We predict the timing of that switch, rather than the direction of Treasury prices.
2. **Does predicting that switch help identify a favorable time to roll?** We compare trading activity and daily execution proxies before, at, and after crossover to assess whether the switch coincides with favorable conditions for replacing the expiring position.

## Data and the migration process

We use **Databento CME Globex `GLBX.MDP3`** daily OHLCV and open-interest statistics, requested through `ZN.FUT` for January 1, 2020–September 30, 2026. Quarterly front/next pairs produce **837 observations**, indexed −25 through +5 around each supplied First Notice Day (FND).

![Volume and open-interest migration across 27 ZN rolls](docs/figures/liquidity_migration.png)

Open-interest crossover precedes volume crossover in **23 of 27 rolls**, coinciding in four. Volume crosses at −2 in 18 rolls, −3 in five, and −1 in four. This separates the movement of positions from the point when next-contract outright trading becomes dominant.

## Why use volume crossover as the target?

We define crossover as the first observation where next-contract volume exceeds **50% of combined front/next outright volume**. The share normalizes activity across rolls, and 50% marks an interpretable change in trading dominance. It is an observable research milestone; it does not identify the cheapest executable trade. OI crossover tracks positioning separately, while calendar-spread activity may better describe the roll itself. Realized-crossover alignment uses hindsight and is descriptive.

## How sharp is the crossover?

We fit each roll's volume and OI shares with a logistic curve, with midpoint τ and steepness k. The fitted 20%–80% width is **2 ln(4)/k**: median widths are approximately **3.1 observations for volume** and **3.5 for OI**. The transition is concentrated, although daily data cannot resolve its intraday shape. These full-window fits describe migration rather than predict it.

![Centered logistic curves and per-roll transition widths](docs/figures/crossover_sharpness.png)

## Feature selection and forecasting methodology

Features represent migration progress, momentum, positioning, and roll activity. We use a small, interpretable set because there are only 27 rolls; additional predictors can fit historical noise.

| Feature | Construction | Economic role |
|---|---|---|
| Next-contract volume share | Next volume / combined outright volume | Progress toward trading dominance |
| Three-observation volume-share change | Current share minus share three observations earlier | Recent migration momentum |
| Five-observation volume-share change | Current share minus share five observations earlier | Broader migration momentum |
| Next-contract OI share | Next OI / combined outright OI | Progress of position migration |
| Three-observation OI-share change | Current OI share minus share three observations earlier | Position-transfer momentum |
| Log spread volume | `log(1 + spread_volume)` | Roll activity, with compressed scale |
| FND-relative index | Recorded panel offset | Calendar timing for probability forecasts |

Date forecasts use one actual snapshot at or before −10 and at least eight earlier, completed rolls. Within training history, features with insufficient observations or zero variance are excluded; missing values are median-imputed and predictors standardized. No outcome-derived “days until crossover” enters predictors. End-of-observation features cannot select execution on that same observation.

We evaluate **three feature sets × five ridge strengths**, fitting each forecast on earlier completed rolls. The −10 and −5 decision cutoffs are evaluated separately on the same 19 later rolls. Model-selection procedures and the full comparison are documented in [timing methodology](docs/issue-20-model-improvements.md).

## Forecast results: accuracy and practical meaning

**One of the best-performing tested ridge models uses just two inputs: the next contract's volume share and its change over the last three observations.** Ridge regularization with **alpha 1,000** limits how strongly the model reacts to these inputs, helping keep forecasts stable in a small sample. The model predicts the observation when the next contract first takes more than half of combined outright trading volume.

| Forecast result | Snapshot at or before −10 | Snapshot at or before −5 |
|---|---:|---:|
| Held-out quarterly rolls | 19 | 19 |
| Exact crossover predictions | **14 / 19 (74%)** | **14 / 19 (74%)** |
| Predictions within one observation | **19 / 19 (100%)** | **19 / 19 (100%)** |
| Mean absolute timing error | **0.263 observations** | **0.263 observations** |
| Largest timing error | **1 observation** | **1 observation** |

These columns evaluate the same 19 rolls at two forecast cutoffs, rather than 38 separate rolls. Results are from the saved [feature-grid summary](docs/tables/feature_grid_summary.csv). This specification is highlighted based on its performance in the exploratory comparison; confirmation on new rolls is still needed.

**What does the accuracy mean?** The model picked the exact crossover observation roughly three times out of four. For the other five rolls, it was one observation early or late. Adding those five one-observation misses and dividing by 19 gives **0.263**. This is an average distance between predicted and actual crossover, measured in recorded observations; it does not mean we can predict within a quarter of a trading day or identify an intraday time.

**What can we do with that forecast?** It provides an advance estimate of when the next contract will become the main venue for outright trading. A trader could use that estimate to plan monitoring and order preparation around a narrow migration window. The execution analysis below then asks whether favorable conditions for the roll occur before, during, or after that window. Timing accuracy alone does not establish trading-cost savings.

Separately, calendar-only and lagged-liquidity logistic models estimate the probability of crossover within one, three, or five observations. Those probability forecasts use Brier score and log loss rather than the date-error measures above.

## Could we measure execution costs instead?

Yes: observed calendar-spread bid/ask quotes and available size would support a more direct execution comparison; depth would be needed to assess larger positions. Our downloaded data contains daily prices, volume, and OI, so this project measures **proxies**, not actual spreads, fills, or slippage.

| Available measure | Definition | What it cannot establish |
|---|---|---|
| Daily half-range | Half of spread high minus low | The executable bid/ask spread |
| Roll-style half-spread | Square root of negative trailing price-change autocovariance; missing otherwise | Observed costs; coverage varies |
| Spread activity share | Spread volume / (front + next + spread volume) | Available quoted size or capacity |
| Price-movement / volume | Absolute spread-price change × 100,000 / spread volume | Causal price impact of a trade |

![Activity and execution proxies around realized volume crossover](docs/figures/crossover_event_conditions.png)

Calendar-spread activity is strongest before crossover, while several proxies deteriorate near or after it. Bands show cross-roll dispersion; observation counts expose incomplete coverage.

![Execution proxies on matching quarterly rolls](docs/figures/execution_conditions.png)

Fixed −5 has the lowest median half-range; −3 has the highest median spread-activity share among tested schedules. Other proxy rankings depend on matching the available rolls. This supports questioning crossover as an execution objective, without establishing an optimal date or cost saving.

## Interpretation and limits

The inherited index includes **140 Sunday-labeled rows**, rather than verified CME sessions. Original event/publication timestamps were discarded, leaving holiday alignment and OI availability unresolved. Durations therefore use **panel observations**. The sample is small and repeatedly explored. Session validation and a limited quote-data study are needed before translating these descriptive findings into an execution recommendation.

## How to reproduce

### 1. Install dependencies

Use Python 3.10+ from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-workflow.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
The workflow dependencies include the model packages and SciPy through
scikit-learn. Raw market data is not committed; a fresh download requires a
Databento account with access to `GLBX.MDP3` and may incur charges.

### 2. Download and prepare the data

```bash
export DATABENTO_API_KEY="your_key_here"
python scripts/run_pipeline.py --dry-run
python scripts/run_pipeline.py
```

On PowerShell, set the key with `$env:DATABENTO_API_KEY="your_key_here"`.
Notebook 01 requests the fixed 2020–2026 window and writes raw Parquet files to
`data/raw/`. It downloads again when rerun. To reuse those files without
calling Databento:

```bash
python scripts/run_pipeline.py --start-at 2
```

The runner executes notebooks 01–06 and `07_predict_crossover.ipynb`, in that
order, with `notebooks/` as the working directory. It creates processed data
and probability-prediction CSVs in `data/processed/`. Notebook outputs are not
saved back by the runner; open and run individual notebooks to view their
inline charts. The download itself has not been repeated for this README revision.

### 3. Reproduce this README's figures and tables

After notebook 02 has produced `data/processed/zn_roll_panel.parquet`:

```bash
python scripts/build_readme_figures.py
```

This offline command rebuilds the standardized panel with
`07_build_standardized_roll_panel.ipynb`, executes
`10_analyze_crossover_geometry_and_costs.ipynb`, and runs the shared timing
comparison and exploratory nested feature grid. It regenerates the six figures in `docs/figures/`
and their supporting CSVs in `docs/tables/`. Intermediate geometry outputs
are in `results/crossover_geometry/`; timing predictions, date audit,
configuration, sensitivity tables, and report are in `results/issue-20/`.
The feature-grid diagnostics and per-roll selections are in
`results/issue-20/grid-search/`; copies of the summary and selections are
included in `docs/tables/` alongside the event-time conditions.
These intermediate directories are git-ignored. The README figures were
regenerated from the cached 27-roll dataset for this revision.

### 4. Explore the remaining analyses

Run all cells of `notebooks/08_test_roll_timing_costs.ipynb` after the pipeline
and standardized-panel step above to compare forecast-timed execution proxies
against benchmark schedules. Notebook 09 displays the timing comparisons and
exploratory extensions; the same calculations are available from the root:

```bash
python -m src.timing_model
python -m src.timing_experiments
python -m src.timing_grid_search
python -m unittest discover -s tests -v
```

See [timing methodology and limitations](docs/issue-20-model-improvements.md)
and [probability-model documentation](docs/issue-7-model.md) for settings and
interpretation. Probability scores and date MAE answer different questions.

### Repository guide

| Path | Contents |
|---|---|
| `notebooks/01`–`06` | Data collection, cleaning, validation, migration, events, and indicators |
| `notebooks/07_predict_crossover.ipynb` | Probability-model evaluation |
| `notebooks/07_build_standardized_roll_panel.ipynb` | Prices and execution proxies |
| `notebooks/08`–`10` | Timing costs, forecast comparisons, and migration geometry |
| `src/` | Reusable prediction and chronological validation code |
| `docs/figures/`, `docs/tables/` | Committed README evidence |
| `scripts/`, `tests/` | Execution helpers and validation |
| `data/`, `results/` | Local market data and generated research outputs |
