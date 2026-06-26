# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-falcons-vs-mouz-bo3-plkh_Ps38mI3o_rFlgAljz/falcons-vs-mouz-m3-nuke-p3.csv`
- round_num: `6`
- rows: `222`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.328522 | 0.172814 | 0.520868 | 0.720721 | 0.671478 |
| xgboost | 0.310627 | 0.152027 | 0.445171 | 0.720721 | 0.689373 |

## Closer Per Tick

- lstm: `86`
- xgboost: `136`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
