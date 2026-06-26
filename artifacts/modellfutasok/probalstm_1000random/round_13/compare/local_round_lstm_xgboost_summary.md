# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `1`
- rows: `104`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.488179 | 0.245457 | 0.683196 | 0.836538 | 0.511821 |
| xgboost | 0.457223 | 0.228807 | 0.645692 | 0.836538 | 0.542777 |

## Closer Per Tick

- lstm: `21`
- xgboost: `83`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
