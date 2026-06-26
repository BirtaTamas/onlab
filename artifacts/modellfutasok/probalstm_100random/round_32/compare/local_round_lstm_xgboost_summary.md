# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `6`
- rows: `158`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.538667 | 0.412791 | 1.105882 | 0.386076 | 0.538667 |
| xgboost | 0.610519 | 0.511037 | 1.603529 | 0.348101 | 0.610519 |

## Closer Per Tick

- lstm: `153`
- xgboost: `5`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
