# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-flyquest-vs-lynn-vision-bo3-tBzyC_GrP1HzVZ3u3bXk3k/flyquest-vs-lynn-vision-m2-anubis.csv`
- round_num: `3`
- rows: `191`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.415658 | 0.197036 | 0.571842 | 0.523560 | 0.415658 |
| xgboost | 0.360343 | 0.141522 | 0.460692 | 0.952880 | 0.360343 |

## Closer Per Tick

- lstm: `49`
- xgboost: `142`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
