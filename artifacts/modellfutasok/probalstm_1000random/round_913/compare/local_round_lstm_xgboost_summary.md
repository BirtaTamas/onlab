# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `11`
- rows: `275`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.194471 | 0.053905 | 0.230774 | 0.974545 | 0.194471 |
| xgboost | 0.269638 | 0.091169 | 0.331814 | 0.960000 | 0.269638 |

## Closer Per Tick

- lstm: `224`
- xgboost: `51`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
