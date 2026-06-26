# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `17`
- rows: `160`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.377631 | 0.154305 | 0.486336 | 1.000000 | 0.377631 |
| xgboost | 0.519940 | 0.285568 | 0.755729 | 0.081250 | 0.519940 |

## Closer Per Tick

- lstm: `160`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
