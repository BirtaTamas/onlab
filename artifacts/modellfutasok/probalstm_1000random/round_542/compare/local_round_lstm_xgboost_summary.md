# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `1`
- rows: `170`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.623522 | 0.405544 | 1.047866 | 0.105882 | 0.376478 |
| xgboost | 0.349784 | 0.151561 | 0.464447 | 0.847059 | 0.650216 |

## Closer Per Tick

- lstm: `0`
- xgboost: `170`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
