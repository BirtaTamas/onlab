# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `14`
- rows: `237`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.221316 | 0.074417 | 0.271385 | 1.000000 | 0.221316 |
| xgboost | 0.264245 | 0.096957 | 0.331584 | 1.000000 | 0.264245 |

## Closer Per Tick

- lstm: `221`
- xgboost: `16`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
