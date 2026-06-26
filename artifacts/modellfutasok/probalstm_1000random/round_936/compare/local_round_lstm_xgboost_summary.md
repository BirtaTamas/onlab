# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-nemiga-vs-m80-bo3-A9YADMgFNfEy-U6IHDyx-U/nemiga-vs-m80-m2-dust2.csv`
- round_num: `8`
- rows: `136`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.192500 | 0.055567 | 0.228723 | 1.000000 | 0.192500 |
| xgboost | 0.315230 | 0.127070 | 0.407708 | 0.830882 | 0.315230 |

## Closer Per Tick

- lstm: `130`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
