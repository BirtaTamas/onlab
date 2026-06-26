# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m1-inferno.csv`
- round_num: `9`
- rows: `235`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.399346 | 0.176505 | 0.541209 | 0.846809 | 0.600654 |
| xgboost | 0.460714 | 0.235280 | 0.667526 | 0.693617 | 0.539286 |

## Closer Per Tick

- lstm: `200`
- xgboost: `35`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
