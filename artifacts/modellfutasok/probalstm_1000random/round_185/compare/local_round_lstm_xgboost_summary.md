# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `6`
- rows: `252`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.470872 | 0.236732 | 0.670253 | 0.698413 | 0.470872 |
| xgboost | 0.503888 | 0.265721 | 0.727450 | 0.285714 | 0.503888 |

## Closer Per Tick

- lstm: `175`
- xgboost: `77`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
