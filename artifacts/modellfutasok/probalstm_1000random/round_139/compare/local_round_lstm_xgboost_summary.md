# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-g2-vs-betboom-bo3-pCfbtiY01aL_JW2Hy1pnZ6/g2-vs-betboom-m1-anubis.csv`
- round_num: `10`
- rows: `171`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.247209 | 0.102470 | 0.323155 | 0.906433 | 0.247209 |
| xgboost | 0.237928 | 0.091963 | 0.303761 | 1.000000 | 0.237928 |

## Closer Per Tick

- lstm: `96`
- xgboost: `75`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
