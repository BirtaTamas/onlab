# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `9`
- rows: `158`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.558411 | 0.346063 | 0.880820 | 0.196203 | 0.558411 |
| xgboost | 0.616715 | 0.410919 | 1.027355 | 0.132911 | 0.616715 |

## Closer Per Tick

- lstm: `149`
- xgboost: `9`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
