# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `17`
- rows: `207`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.600535 | 0.397138 | 0.993200 | 0.149758 | 0.399465 |
| xgboost | 0.482618 | 0.260419 | 0.695918 | 0.159420 | 0.517382 |

## Closer Per Tick

- lstm: `0`
- xgboost: `207`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
