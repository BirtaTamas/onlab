# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `6`
- rows: `193`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.076932 | 0.010675 | 0.082918 | 1.000000 | 0.076932 |
| xgboost | 0.162393 | 0.030285 | 0.179938 | 1.000000 | 0.162393 |

## Closer Per Tick

- lstm: `193`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
