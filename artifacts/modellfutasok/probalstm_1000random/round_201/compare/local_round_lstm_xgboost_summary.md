# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `9`
- rows: `198`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.367673 | 0.251474 | 0.633892 | 0.479798 | 0.367673 |
| xgboost | 0.376023 | 0.243682 | 0.638044 | 0.479798 | 0.376023 |

## Closer Per Tick

- lstm: `127`
- xgboost: `71`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
