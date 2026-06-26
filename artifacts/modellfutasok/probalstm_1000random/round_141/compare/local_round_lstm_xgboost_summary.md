# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `17`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.277240 | 0.115754 | 0.362172 | 1.000000 | 0.722760 |
| xgboost | 0.247026 | 0.103196 | 0.322004 | 1.000000 | 0.752974 |

## Closer Per Tick

- lstm: `57`
- xgboost: `173`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
