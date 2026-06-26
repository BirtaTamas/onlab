# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `7`
- rows: `148`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.325148 | 0.138533 | 0.430216 | 0.804054 | 0.674852 |
| xgboost | 0.298372 | 0.142986 | 0.418407 | 0.682432 | 0.701628 |

## Closer Per Tick

- lstm: `55`
- xgboost: `93`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
