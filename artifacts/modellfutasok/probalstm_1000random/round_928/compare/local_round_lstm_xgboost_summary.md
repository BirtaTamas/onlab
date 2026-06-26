# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `17`
- rows: `201`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.365321 | 0.155975 | 0.480135 | 0.955224 | 0.365321 |
| xgboost | 0.447519 | 0.215801 | 0.614006 | 0.507463 | 0.447519 |

## Closer Per Tick

- lstm: `191`
- xgboost: `10`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
