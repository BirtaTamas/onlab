# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `17`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.382272 | 0.189643 | 0.538466 | 0.604348 | 0.617728 |
| xgboost | 0.304336 | 0.121124 | 0.390565 | 1.000000 | 0.695664 |

## Closer Per Tick

- lstm: `19`
- xgboost: `211`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
