# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `1`
- rows: `102`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.319345 | 0.118109 | 0.402849 | 0.980392 | 0.680655 |
| xgboost | 0.266550 | 0.093898 | 0.333692 | 1.000000 | 0.733450 |

## Closer Per Tick

- lstm: `26`
- xgboost: `76`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
