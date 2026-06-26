# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `3`
- rows: `161`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.044947 | 0.009598 | 0.050819 | 1.000000 | 0.044947 |
| xgboost | 0.068170 | 0.016939 | 0.078763 | 1.000000 | 0.068170 |

## Closer Per Tick

- lstm: `159`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
