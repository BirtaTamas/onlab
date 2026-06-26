# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m2-mirage.csv`
- round_num: `14`
- rows: `123`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.340207 | 0.160842 | 0.504857 | 0.829268 | 0.340207 |
| xgboost | 0.372019 | 0.169680 | 0.515247 | 0.853659 | 0.372019 |

## Closer Per Tick

- lstm: `92`
- xgboost: `31`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
