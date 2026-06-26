# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `9`
- rows: `233`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.420434 | 0.213909 | 0.591121 | 0.381974 | 0.420434 |
| xgboost | 0.400742 | 0.195569 | 0.554743 | 0.369099 | 0.400742 |

## Closer Per Tick

- lstm: `88`
- xgboost: `145`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
