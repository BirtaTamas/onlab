# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `2`
- rows: `153`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.098766 | 0.040603 | 0.132316 | 0.934641 | 0.098766 |
| xgboost | 0.121458 | 0.042577 | 0.152917 | 0.954248 | 0.121458 |

## Closer Per Tick

- lstm: `136`
- xgboost: `17`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
