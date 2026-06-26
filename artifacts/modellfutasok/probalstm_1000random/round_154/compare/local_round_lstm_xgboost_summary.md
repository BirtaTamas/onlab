# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `15`
- rows: `123`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.080139 | 0.007482 | 0.084165 | 1.000000 | 0.919861 |
| xgboost | 0.029976 | 0.001029 | 0.030503 | 1.000000 | 0.970024 |

## Closer Per Tick

- lstm: `0`
- xgboost: `123`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
