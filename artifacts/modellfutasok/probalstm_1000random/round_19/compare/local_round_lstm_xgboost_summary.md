# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `2`
- rows: `208`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.647957 | 0.504272 | 1.288587 | 0.216346 | 0.352043 |
| xgboost | 0.542272 | 0.358683 | 0.894900 | 0.216346 | 0.457728 |

## Closer Per Tick

- lstm: `0`
- xgboost: `208`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
