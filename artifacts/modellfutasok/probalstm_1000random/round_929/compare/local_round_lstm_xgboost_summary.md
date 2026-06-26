# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `13`
- rows: `249`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.297325 | 0.140962 | 0.406674 | 0.542169 | 0.297325 |
| xgboost | 0.385719 | 0.209563 | 0.564030 | 0.465863 | 0.385719 |

## Closer Per Tick

- lstm: `226`
- xgboost: `23`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
