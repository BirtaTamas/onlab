# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `2`
- rows: `137`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.257299 | 0.077310 | 0.307311 | 1.000000 | 0.257299 |
| xgboost | 0.251372 | 0.068636 | 0.293965 | 1.000000 | 0.251372 |

## Closer Per Tick

- lstm: `62`
- xgboost: `75`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
