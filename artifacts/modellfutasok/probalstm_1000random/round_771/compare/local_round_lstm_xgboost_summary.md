# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m1-mirage.csv`
- round_num: `4`
- rows: `117`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.345195 | 0.186956 | 0.521091 | 0.529915 | 0.345195 |
| xgboost | 0.471936 | 0.278357 | 0.784524 | 0.435897 | 0.471936 |

## Closer Per Tick

- lstm: `102`
- xgboost: `15`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
