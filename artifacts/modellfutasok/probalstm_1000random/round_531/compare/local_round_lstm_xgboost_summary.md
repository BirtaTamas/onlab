# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `18`
- rows: `295`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.142466 | 0.049916 | 0.177968 | 0.962712 | 0.142466 |
| xgboost | 0.165776 | 0.055988 | 0.204763 | 1.000000 | 0.165776 |

## Closer Per Tick

- lstm: `247`
- xgboost: `48`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
