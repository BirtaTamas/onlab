# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `2`
- rows: `189`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.443824 | 0.227560 | 0.630953 | 0.423280 | 0.443824 |
| xgboost | 0.544259 | 0.310053 | 0.815991 | 0.179894 | 0.544259 |

## Closer Per Tick

- lstm: `168`
- xgboost: `21`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
