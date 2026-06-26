# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `6`
- rows: `162`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.338828 | 0.124956 | 0.424258 | 1.000000 | 0.661172 |
| xgboost | 0.273738 | 0.087182 | 0.331790 | 1.000000 | 0.726262 |

## Closer Per Tick

- lstm: `16`
- xgboost: `146`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
