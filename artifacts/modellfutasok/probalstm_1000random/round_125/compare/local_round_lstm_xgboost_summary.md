# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `4`
- rows: `123`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.444964 | 0.267328 | 0.725469 | 0.422764 | 0.555036 |
| xgboost | 0.383048 | 0.201723 | 0.556185 | 0.552846 | 0.616952 |

## Closer Per Tick

- lstm: `7`
- xgboost: `116`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
