# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `12`
- rows: `194`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.087953 | 0.033495 | 0.112850 | 0.979381 | 0.087953 |
| xgboost | 0.113527 | 0.044949 | 0.149339 | 0.855670 | 0.113527 |

## Closer Per Tick

- lstm: `192`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
