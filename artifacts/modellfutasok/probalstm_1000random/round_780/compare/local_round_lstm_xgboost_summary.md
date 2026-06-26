# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `15`
- rows: `204`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.546999 | 0.346135 | 0.932115 | 0.514706 | 0.546999 |
| xgboost | 0.549133 | 0.348245 | 0.932286 | 0.416667 | 0.549133 |

## Closer Per Tick

- lstm: `89`
- xgboost: `115`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
