# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `11`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.221027 | 0.095037 | 0.298053 | 0.803468 | 0.221027 |
| xgboost | 0.251829 | 0.093313 | 0.319276 | 0.838150 | 0.251829 |

## Closer Per Tick

- lstm: `121`
- xgboost: `52`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
