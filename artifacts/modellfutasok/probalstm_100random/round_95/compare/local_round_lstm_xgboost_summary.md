# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `1`
- rows: `124`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.392812 | 0.199071 | 0.548489 | 0.370968 | 0.392812 |
| xgboost | 0.365062 | 0.169856 | 0.492261 | 0.959677 | 0.365062 |

## Closer Per Tick

- lstm: `27`
- xgboost: `97`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
