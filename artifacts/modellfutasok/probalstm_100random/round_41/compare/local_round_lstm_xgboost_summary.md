# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-nrg-dust2-QDtqFlW1Z9UhZpBNOAavnd/heroic-vs-nrg-dust2.csv`
- round_num: `1`
- rows: `138`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.259808 | 0.100832 | 0.335188 | 0.840580 | 0.259808 |
| xgboost | 0.335034 | 0.149097 | 0.457347 | 0.768116 | 0.335034 |

## Closer Per Tick

- lstm: `125`
- xgboost: `13`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
