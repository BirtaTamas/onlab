# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m3-nuke.csv`
- round_num: `7`
- rows: `196`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.346655 | 0.138491 | 0.446096 | 0.908163 | 0.653345 |
| xgboost | 0.277326 | 0.096974 | 0.344796 | 0.943878 | 0.722674 |

## Closer Per Tick

- lstm: `34`
- xgboost: `162`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
