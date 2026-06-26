# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `11`
- rows: `109`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.234216 | 0.066432 | 0.276472 | 1.000000 | 0.765784 |
| xgboost | 0.276554 | 0.099752 | 0.346116 | 1.000000 | 0.723446 |

## Closer Per Tick

- lstm: `77`
- xgboost: `32`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
