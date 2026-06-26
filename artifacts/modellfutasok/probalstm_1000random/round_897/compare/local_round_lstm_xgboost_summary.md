# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `2`
- rows: `244`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.136894 | 0.024622 | 0.151538 | 1.000000 | 0.863106 |
| xgboost | 0.093516 | 0.011767 | 0.100131 | 1.000000 | 0.906484 |

## Closer Per Tick

- lstm: `34`
- xgboost: `210`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
