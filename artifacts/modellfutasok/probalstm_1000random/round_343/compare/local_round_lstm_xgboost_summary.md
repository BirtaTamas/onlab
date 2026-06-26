# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `6`
- rows: `235`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.212705 | 0.055492 | 0.248358 | 0.991489 | 0.212705 |
| xgboost | 0.296079 | 0.104984 | 0.374665 | 0.914894 | 0.296079 |

## Closer Per Tick

- lstm: `216`
- xgboost: `19`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
