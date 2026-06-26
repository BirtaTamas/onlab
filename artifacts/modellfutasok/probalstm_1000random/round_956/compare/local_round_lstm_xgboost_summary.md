# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `5`
- rows: `269`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.573407 | 0.363484 | 1.032498 | 0.371747 | 0.426593 |
| xgboost | 0.521693 | 0.303243 | 0.842188 | 0.546468 | 0.478307 |

## Closer Per Tick

- lstm: `44`
- xgboost: `225`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
