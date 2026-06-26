# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `12`
- rows: `211`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.214566 | 0.079163 | 0.271415 | 0.886256 | 0.214566 |
| xgboost | 0.260209 | 0.087118 | 0.318470 | 1.000000 | 0.260209 |

## Closer Per Tick

- lstm: `150`
- xgboost: `61`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
