# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `5`
- rows: `217`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.446263 | 0.242147 | 0.675748 | 0.700461 | 0.553737 |
| xgboost | 0.388335 | 0.199783 | 0.568448 | 0.792627 | 0.611665 |

## Closer Per Tick

- lstm: `70`
- xgboost: `147`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
