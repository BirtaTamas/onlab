# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m2-inferno.csv`
- round_num: `12`
- rows: `170`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.400074 | 0.175807 | 0.536266 | 0.735294 | 0.599926 |
| xgboost | 0.597254 | 0.402496 | 1.088836 | 0.429412 | 0.402746 |

## Closer Per Tick

- lstm: `158`
- xgboost: `12`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
