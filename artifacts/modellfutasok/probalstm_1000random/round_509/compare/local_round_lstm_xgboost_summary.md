# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `5`
- rows: `218`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.063916 | 0.012460 | 0.071303 | 1.000000 | 0.063916 |
| xgboost | 0.119789 | 0.034594 | 0.141751 | 1.000000 | 0.119789 |

## Closer Per Tick

- lstm: `218`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
