# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `24`
- rows: `182`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.367286 | 0.213162 | 0.646132 | 0.807692 | 0.367286 |
| xgboost | 0.495974 | 0.305749 | 0.963768 | 0.351648 | 0.495974 |

## Closer Per Tick

- lstm: `181`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
