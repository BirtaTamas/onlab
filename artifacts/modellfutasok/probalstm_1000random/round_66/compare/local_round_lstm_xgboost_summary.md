# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-spirit-vs-virtuspro-bo3-NVE3FTuEWJ64hP6AT-Vo9S/spirit-vs-virtus-pro-m2-overpass.csv`
- round_num: `11`
- rows: `226`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.501146 | 0.285006 | 0.807198 | 0.517699 | 0.498854 |
| xgboost | 0.480729 | 0.262494 | 0.730290 | 0.561947 | 0.519271 |

## Closer Per Tick

- lstm: `94`
- xgboost: `132`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
