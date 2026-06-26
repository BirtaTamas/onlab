# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `2`
- rows: `165`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.274985 | 0.084272 | 0.329295 | 1.000000 | 0.725015 |
| xgboost | 0.251570 | 0.071157 | 0.296487 | 1.000000 | 0.748430 |

## Closer Per Tick

- lstm: `74`
- xgboost: `91`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
