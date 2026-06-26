# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `14`
- rows: `175`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.311006 | 0.131898 | 0.406022 | 0.988571 | 0.688994 |
| xgboost | 0.236984 | 0.080624 | 0.290196 | 1.000000 | 0.763016 |

## Closer Per Tick

- lstm: `0`
- xgboost: `175`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
