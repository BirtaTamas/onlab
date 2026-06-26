# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-spirit-vs-virtuspro-bo3-NVE3FTuEWJ64hP6AT-Vo9S/spirit-vs-virtus-pro-m2-overpass.csv`
- round_num: `10`
- rows: `172`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.329974 | 0.125170 | 0.429444 | 0.912791 | 0.670026 |
| xgboost | 0.399545 | 0.170521 | 0.525655 | 0.901163 | 0.600455 |

## Closer Per Tick

- lstm: `146`
- xgboost: `26`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
