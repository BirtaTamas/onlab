# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-flyquest-vs-lynn-vision-bo3-tBzyC_GrP1HzVZ3u3bXk3k/flyquest-vs-lynn-vision-m2-anubis.csv`
- round_num: `8`
- rows: `226`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.474849 | 0.260455 | 0.775766 | 0.747788 | 0.525151 |
| xgboost | 0.486724 | 0.287616 | 0.832366 | 0.707965 | 0.513276 |

## Closer Per Tick

- lstm: `155`
- xgboost: `71`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
