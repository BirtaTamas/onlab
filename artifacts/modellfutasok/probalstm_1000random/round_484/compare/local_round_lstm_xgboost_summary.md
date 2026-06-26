# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `16`
- rows: `181`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.180146 | 0.055767 | 0.217652 | 1.000000 | 0.180146 |
| xgboost | 0.286968 | 0.112684 | 0.366661 | 0.928177 | 0.286968 |

## Closer Per Tick

- lstm: `181`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
