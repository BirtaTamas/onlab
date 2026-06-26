# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m2-mirage.csv`
- round_num: `1`
- rows: `147`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.256545 | 0.125766 | 0.358703 | 0.619048 | 0.256545 |
| xgboost | 0.318702 | 0.171556 | 0.478122 | 0.523810 | 0.318702 |

## Closer Per Tick

- lstm: `147`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
