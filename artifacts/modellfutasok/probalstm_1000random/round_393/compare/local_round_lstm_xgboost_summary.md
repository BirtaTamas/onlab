# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `3`
- rows: `217`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.461476 | 0.222471 | 0.633050 | 0.834101 | 0.538524 |
| xgboost | 0.517913 | 0.286379 | 0.764410 | 0.285714 | 0.482087 |

## Closer Per Tick

- lstm: `173`
- xgboost: `44`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
