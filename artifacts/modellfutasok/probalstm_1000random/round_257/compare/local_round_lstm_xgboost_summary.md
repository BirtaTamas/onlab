# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `2`
- rows: `227`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.605563 | 0.505935 | 1.415011 | 0.352423 | 0.605563 |
| xgboost | 0.718578 | 0.640090 | 2.297610 | 0.273128 | 0.718578 |

## Closer Per Tick

- lstm: `226`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
