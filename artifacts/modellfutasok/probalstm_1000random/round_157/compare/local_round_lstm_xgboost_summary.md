# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `5`
- rows: `282`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.323132 | 0.158996 | 0.447115 | 0.719858 | 0.323132 |
| xgboost | 0.401599 | 0.241620 | 0.630461 | 0.446809 | 0.401599 |

## Closer Per Tick

- lstm: `276`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
