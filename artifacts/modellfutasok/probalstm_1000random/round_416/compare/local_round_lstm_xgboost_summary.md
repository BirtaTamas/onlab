# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `20`
- rows: `179`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.380932 | 0.189432 | 0.550201 | 0.614525 | 0.380932 |
| xgboost | 0.562042 | 0.337761 | 0.887305 | 0.234637 | 0.562042 |

## Closer Per Tick

- lstm: `175`
- xgboost: `4`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
