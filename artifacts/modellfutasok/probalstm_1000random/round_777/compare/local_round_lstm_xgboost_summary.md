# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `11`
- rows: `214`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.547802 | 0.360589 | 1.104481 | 0.621495 | 0.452198 |
| xgboost | 0.520117 | 0.318688 | 0.891491 | 0.415888 | 0.479883 |

## Closer Per Tick

- lstm: `97`
- xgboost: `117`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
