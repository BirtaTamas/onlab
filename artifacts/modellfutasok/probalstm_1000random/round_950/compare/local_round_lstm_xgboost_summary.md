# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `10`
- rows: `188`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.101189 | 0.041422 | 0.132245 | 0.978723 | 0.101189 |
| xgboost | 0.124813 | 0.049901 | 0.163451 | 0.829787 | 0.124813 |

## Closer Per Tick

- lstm: `188`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
