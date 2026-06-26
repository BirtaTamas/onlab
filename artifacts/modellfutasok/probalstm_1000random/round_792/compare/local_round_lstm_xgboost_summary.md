# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `10`
- rows: `217`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.052698 | 0.006858 | 0.056613 | 1.000000 | 0.947302 |
| xgboost | 0.011639 | 0.000429 | 0.011859 | 1.000000 | 0.988361 |

## Closer Per Tick

- lstm: `0`
- xgboost: `217`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
