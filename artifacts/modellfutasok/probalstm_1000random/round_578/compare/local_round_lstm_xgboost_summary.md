# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `2`
- rows: `149`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.406271 | 0.210718 | 0.581625 | 0.496644 | 0.406271 |
| xgboost | 0.485758 | 0.288467 | 0.767942 | 0.476510 | 0.485758 |

## Closer Per Tick

- lstm: `129`
- xgboost: `20`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
