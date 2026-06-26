# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `17`
- rows: `244`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.630979 | 0.421237 | 1.178824 | 0.032787 | 0.630979 |
| xgboost | 0.690185 | 0.513995 | 1.652576 | 0.241803 | 0.690185 |

## Closer Per Tick

- lstm: `179`
- xgboost: `65`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
