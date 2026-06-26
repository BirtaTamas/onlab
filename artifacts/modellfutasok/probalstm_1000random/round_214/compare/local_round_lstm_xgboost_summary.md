# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-3dmax-vs-vitality-nuke-h8drweGjLe5Dwjfuh5VfUb/3dmax-vs-vitality-nuke.csv`
- round_num: `3`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.588883 | 0.376514 | 1.035180 | 0.506944 | 0.411117 |
| xgboost | 0.542100 | 0.333223 | 0.909365 | 0.298611 | 0.457900 |

## Closer Per Tick

- lstm: `86`
- xgboost: `58`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
