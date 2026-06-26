# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-falcons-vs-mouz-bo3-plkh_Ps38mI3o_rFlgAljz/falcons-vs-mouz-m3-nuke-p3.csv`
- round_num: `1`
- rows: `184`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.213423 | 0.054299 | 0.246789 | 1.000000 | 0.213423 |
| xgboost | 0.310225 | 0.100646 | 0.377200 | 0.940217 | 0.310225 |

## Closer Per Tick

- lstm: `153`
- xgboost: `31`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
