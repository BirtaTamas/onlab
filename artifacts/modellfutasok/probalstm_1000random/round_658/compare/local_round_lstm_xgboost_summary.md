# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `18`
- rows: `168`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.041999 | 0.004075 | 0.044192 | 1.000000 | 0.041999 |
| xgboost | 0.131345 | 0.034185 | 0.152344 | 1.000000 | 0.131345 |

## Closer Per Tick

- lstm: `168`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
