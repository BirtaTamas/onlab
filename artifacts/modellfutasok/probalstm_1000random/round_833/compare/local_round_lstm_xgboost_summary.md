# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `19`
- rows: `259`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.259546 | 0.123067 | 0.354150 | 0.694981 | 0.259546 |
| xgboost | 0.291346 | 0.137716 | 0.397860 | 0.822394 | 0.291346 |

## Closer Per Tick

- lstm: `192`
- xgboost: `67`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
