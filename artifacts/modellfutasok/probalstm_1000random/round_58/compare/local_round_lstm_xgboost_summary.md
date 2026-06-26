# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `5`
- rows: `303`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.392843 | 0.192868 | 0.545129 | 0.679868 | 0.392843 |
| xgboost | 0.408421 | 0.208019 | 0.580656 | 0.884488 | 0.408421 |

## Closer Per Tick

- lstm: `189`
- xgboost: `114`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
