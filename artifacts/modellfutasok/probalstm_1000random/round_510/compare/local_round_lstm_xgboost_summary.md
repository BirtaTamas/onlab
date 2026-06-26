# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `9`
- rows: `194`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.406688 | 0.224244 | 0.599174 | 0.371134 | 0.593312 |
| xgboost | 0.357833 | 0.182722 | 0.505179 | 0.551546 | 0.642167 |

## Closer Per Tick

- lstm: `15`
- xgboost: `179`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
