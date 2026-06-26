# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `18`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.641463 | 0.458607 | 1.420849 | 0.547826 | 0.358537 |
| xgboost | 0.661925 | 0.475011 | 1.364083 | 0.443478 | 0.338075 |

## Closer Per Tick

- lstm: `168`
- xgboost: `62`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
