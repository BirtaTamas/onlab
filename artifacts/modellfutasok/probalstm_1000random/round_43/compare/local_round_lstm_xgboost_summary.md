# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-metizport-mirage-uJE2h4ym3PvBPopNN8-YOA/tyloo-vs-metizport-mirage.csv`
- round_num: `3`
- rows: `171`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.010700 | 0.000280 | 0.010843 | 1.000000 | 0.010700 |
| xgboost | 0.052491 | 0.005595 | 0.055535 | 1.000000 | 0.052491 |

## Closer Per Tick

- lstm: `171`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
