# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `13`
- rows: `132`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.282929 | 0.129194 | 0.380557 | 1.000000 | 0.717071 |
| xgboost | 0.270584 | 0.127923 | 0.367735 | 0.984848 | 0.729416 |

## Closer Per Tick

- lstm: `48`
- xgboost: `84`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
