# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-big-vs-pain-bo3-So89pkF9idYLRaqhIPbo1H/big-vs-pain-m3-inferno-p3.csv`
- round_num: `10`
- rows: `169`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.537322 | 0.313481 | 0.823789 | 0.218935 | 0.462678 |
| xgboost | 0.536602 | 0.332756 | 0.867172 | 0.213018 | 0.463398 |

## Closer Per Tick

- lstm: `82`
- xgboost: `87`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
