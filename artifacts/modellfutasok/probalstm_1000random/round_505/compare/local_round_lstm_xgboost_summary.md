# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `7`
- rows: `257`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.270522 | 0.124681 | 0.363573 | 0.894942 | 0.270522 |
| xgboost | 0.366475 | 0.211152 | 0.553419 | 0.498054 | 0.366475 |

## Closer Per Tick

- lstm: `256`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
