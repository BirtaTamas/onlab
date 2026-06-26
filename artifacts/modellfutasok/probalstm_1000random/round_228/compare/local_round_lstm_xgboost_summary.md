# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `11`
- rows: `255`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.459360 | 0.243843 | 0.683627 | 0.635294 | 0.540640 |
| xgboost | 0.301835 | 0.111741 | 0.382147 | 0.878431 | 0.698165 |

## Closer Per Tick

- lstm: `9`
- xgboost: `246`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
