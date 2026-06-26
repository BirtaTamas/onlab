# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `2`
- rows: `243`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.431048 | 0.213111 | 0.631817 | 0.876543 | 0.568952 |
| xgboost | 0.358258 | 0.146703 | 0.468285 | 0.934156 | 0.641742 |

## Closer Per Tick

- lstm: `9`
- xgboost: `234`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
