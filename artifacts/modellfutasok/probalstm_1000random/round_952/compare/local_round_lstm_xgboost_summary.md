# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `6`
- rows: `185`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.259903 | 0.125330 | 0.359386 | 0.654054 | 0.259903 |
| xgboost | 0.261979 | 0.130031 | 0.368480 | 0.654054 | 0.261979 |

## Closer Per Tick

- lstm: `133`
- xgboost: `52`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
