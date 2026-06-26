# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-mouz-bo3-D4mE8XcULbH9iT3IhMhdJY/legacy-vs-mouz-m1-ancient.csv`
- round_num: `1`
- rows: `137`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.281623 | 0.113619 | 0.362872 | 0.934307 | 0.281623 |
| xgboost | 0.379826 | 0.176974 | 0.514577 | 0.978102 | 0.379826 |

## Closer Per Tick

- lstm: `124`
- xgboost: `13`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
