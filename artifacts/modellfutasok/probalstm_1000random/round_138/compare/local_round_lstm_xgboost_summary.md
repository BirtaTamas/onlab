# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `6`
- rows: `278`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.288770 | 0.166468 | 0.437758 | 0.579137 | 0.288770 |
| xgboost | 0.191755 | 0.051011 | 0.223706 | 1.000000 | 0.191755 |

## Closer Per Tick

- lstm: `149`
- xgboost: `129`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
