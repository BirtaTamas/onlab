# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `5`
- rows: `277`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.168421 | 0.051886 | 0.202831 | 0.989170 | 0.168421 |
| xgboost | 0.196336 | 0.060594 | 0.236028 | 0.992780 | 0.196336 |

## Closer Per Tick

- lstm: `210`
- xgboost: `67`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
