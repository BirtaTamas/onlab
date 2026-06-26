# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `3`
- rows: `109`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.400395 | 0.205527 | 0.566852 | 0.422018 | 0.400395 |
| xgboost | 0.371964 | 0.177683 | 0.511112 | 0.550459 | 0.371964 |

## Closer Per Tick

- lstm: `52`
- xgboost: `57`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
