# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `12`
- rows: `158`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.440306 | 0.210030 | 0.607115 | 0.487342 | 0.440306 |
| xgboost | 0.505964 | 0.271712 | 0.744264 | 0.405063 | 0.505964 |

## Closer Per Tick

- lstm: `146`
- xgboost: `12`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
