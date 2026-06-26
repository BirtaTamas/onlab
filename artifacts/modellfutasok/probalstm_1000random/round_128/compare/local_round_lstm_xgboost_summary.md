# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-b8-bo3-rUWlZLFFckLiQv1C1wSlHb/g2-vs-b8-m3-ancient.csv`
- round_num: `7`
- rows: `291`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.443960 | 0.270015 | 0.690201 | 0.292096 | 0.443960 |
| xgboost | 0.411878 | 0.224628 | 0.599735 | 0.422680 | 0.411878 |

## Closer Per Tick

- lstm: `94`
- xgboost: `197`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
