# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `9`
- rows: `188`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.158198 | 0.074314 | 0.218449 | 0.877660 | 0.158198 |
| xgboost | 0.220980 | 0.096588 | 0.297794 | 0.723404 | 0.220980 |

## Closer Per Tick

- lstm: `182`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
