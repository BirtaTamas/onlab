# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `14`
- rows: `260`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.597190 | 0.413452 | 1.055597 | 0.265385 | 0.402810 |
| xgboost | 0.574719 | 0.399435 | 1.016521 | 0.276923 | 0.425281 |

## Closer Per Tick

- lstm: `100`
- xgboost: `160`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
