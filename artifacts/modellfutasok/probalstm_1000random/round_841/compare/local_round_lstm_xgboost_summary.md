# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m2-inferno.csv`
- round_num: `9`
- rows: `187`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.432819 | 0.196204 | 0.580243 | 0.780749 | 0.567181 |
| xgboost | 0.382681 | 0.160007 | 0.500465 | 0.887701 | 0.617319 |

## Closer Per Tick

- lstm: `47`
- xgboost: `140`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
