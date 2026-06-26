# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `2`
- rows: `141`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.068831 | 0.005184 | 0.071573 | 1.000000 | 0.931169 |
| xgboost | 0.018540 | 0.000351 | 0.018718 | 1.000000 | 0.981460 |

## Closer Per Tick

- lstm: `0`
- xgboost: `141`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
