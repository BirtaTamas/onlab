# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `9`
- rows: `160`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.363959 | 0.177592 | 0.504128 | 0.631250 | 0.363959 |
| xgboost | 0.464749 | 0.274171 | 0.715142 | 0.281250 | 0.464749 |

## Closer Per Tick

- lstm: `158`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
