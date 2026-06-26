# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `10`
- rows: `151`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.337364 | 0.141064 | 0.443520 | 0.754967 | 0.662636 |
| xgboost | 0.360945 | 0.175755 | 0.505598 | 0.629139 | 0.639055 |

## Closer Per Tick

- lstm: `99`
- xgboost: `52`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
