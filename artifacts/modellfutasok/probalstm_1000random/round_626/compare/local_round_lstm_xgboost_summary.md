# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-big-vs-pain-bo3-So89pkF9idYLRaqhIPbo1H/big-vs-pain-m3-inferno-p3.csv`
- round_num: `4`
- rows: `265`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.244937 | 0.102463 | 0.320082 | 0.920755 | 0.244937 |
| xgboost | 0.259719 | 0.108355 | 0.339801 | 0.807547 | 0.259719 |

## Closer Per Tick

- lstm: `192`
- xgboost: `73`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
