# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `8`
- rows: `190`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.405933 | 0.231321 | 0.615812 | 0.426316 | 0.405933 |
| xgboost | 0.427314 | 0.241221 | 0.654596 | 0.352632 | 0.427314 |

## Closer Per Tick

- lstm: `120`
- xgboost: `70`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
