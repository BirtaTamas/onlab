# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `13`
- rows: `139`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.482212 | 0.266453 | 0.788992 | 0.769784 | 0.517788 |
| xgboost | 0.304906 | 0.146268 | 0.437045 | 0.705036 | 0.695094 |

## Closer Per Tick

- lstm: `34`
- xgboost: `105`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
