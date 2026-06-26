# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `5`
- rows: `185`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.607709 | 0.408387 | 1.056755 | 0.372973 | 0.392291 |
| xgboost | 0.532959 | 0.306724 | 0.804144 | 0.383784 | 0.467041 |

## Closer Per Tick

- lstm: `28`
- xgboost: `157`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
