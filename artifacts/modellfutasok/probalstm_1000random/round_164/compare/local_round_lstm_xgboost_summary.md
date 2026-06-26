# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m3-mirage.csv`
- round_num: `10`
- rows: `293`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.623441 | 0.426830 | 1.078780 | 0.194539 | 0.623441 |
| xgboost | 0.644252 | 0.441817 | 1.118908 | 0.160410 | 0.644252 |

## Closer Per Tick

- lstm: `133`
- xgboost: `160`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
