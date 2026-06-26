# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `16`
- rows: `148`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.243933 | 0.088161 | 0.305681 | 0.979730 | 0.756067 |
| xgboost | 0.235247 | 0.094611 | 0.304381 | 0.932432 | 0.764753 |

## Closer Per Tick

- lstm: `54`
- xgboost: `94`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
