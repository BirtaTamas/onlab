# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `13`
- rows: `199`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.333560 | 0.169634 | 0.524492 | 0.839196 | 0.666440 |
| xgboost | 0.230602 | 0.117793 | 0.340954 | 0.768844 | 0.769398 |

## Closer Per Tick

- lstm: `46`
- xgboost: `153`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
