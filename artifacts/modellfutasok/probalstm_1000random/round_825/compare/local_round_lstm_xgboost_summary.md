# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `6`
- rows: `263`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.508156 | 0.298163 | 0.795471 | 0.220532 | 0.491844 |
| xgboost | 0.414177 | 0.203343 | 0.576488 | 0.768061 | 0.585823 |

## Closer Per Tick

- lstm: `0`
- xgboost: `263`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
