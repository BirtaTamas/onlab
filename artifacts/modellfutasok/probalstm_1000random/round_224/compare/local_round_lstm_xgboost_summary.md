# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m1-dust2.csv`
- round_num: `16`
- rows: `153`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.360130 | 0.152326 | 0.475435 | 0.849673 | 0.360130 |
| xgboost | 0.401009 | 0.180894 | 0.541630 | 0.745098 | 0.401009 |

## Closer Per Tick

- lstm: `119`
- xgboost: `34`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
