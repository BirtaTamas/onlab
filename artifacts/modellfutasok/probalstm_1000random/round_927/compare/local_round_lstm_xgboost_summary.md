# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `3`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.009848 | 0.000159 | 0.009929 | 1.000000 | 0.009848 |
| xgboost | 0.026218 | 0.001117 | 0.026796 | 1.000000 | 0.026218 |

## Closer Per Tick

- lstm: `227`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
