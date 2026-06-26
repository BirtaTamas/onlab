# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `5`
- rows: `290`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.216529 | 0.076633 | 0.270280 | 0.951724 | 0.216529 |
| xgboost | 0.291160 | 0.127498 | 0.386039 | 0.951724 | 0.291160 |

## Closer Per Tick

- lstm: `274`
- xgboost: `16`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
