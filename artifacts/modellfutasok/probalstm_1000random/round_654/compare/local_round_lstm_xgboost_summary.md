# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `5`
- rows: `142`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.273384 | 0.211607 | 0.572624 | 0.704225 | 0.273384 |
| xgboost | 0.329972 | 0.279938 | 0.923248 | 0.690141 | 0.329972 |

## Closer Per Tick

- lstm: `119`
- xgboost: `23`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
