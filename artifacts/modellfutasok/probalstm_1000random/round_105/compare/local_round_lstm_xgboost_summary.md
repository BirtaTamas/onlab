# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `11`
- rows: `249`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.349849 | 0.203001 | 0.529266 | 0.510040 | 0.349849 |
| xgboost | 0.409617 | 0.281180 | 0.714266 | 0.449799 | 0.409617 |

## Closer Per Tick

- lstm: `236`
- xgboost: `13`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
