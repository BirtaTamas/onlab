# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `16`
- rows: `135`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.469431 | 0.230001 | 0.658005 | 0.681481 | 0.530569 |
| xgboost | 0.460984 | 0.222588 | 0.637654 | 0.911111 | 0.539016 |

## Closer Per Tick

- lstm: `50`
- xgboost: `85`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
