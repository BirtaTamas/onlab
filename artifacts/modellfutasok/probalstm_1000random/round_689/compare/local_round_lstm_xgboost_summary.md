# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `1`
- rows: `181`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.239610 | 0.105166 | 0.321381 | 0.635359 | 0.239610 |
| xgboost | 0.248761 | 0.096627 | 0.320092 | 0.994475 | 0.248761 |

## Closer Per Tick

- lstm: `101`
- xgboost: `80`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
