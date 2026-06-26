# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-heroic-vs-aurora-bo3-QigxwcikBDdlIOkrYDpY7y/heroic-vs-aurora-m2-dust2.csv`
- round_num: `21`
- rows: `140`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.129420 | 0.057111 | 0.176618 | 0.850000 | 0.129420 |
| xgboost | 0.122474 | 0.046305 | 0.157212 | 0.850000 | 0.122474 |

## Closer Per Tick

- lstm: `35`
- xgboost: `105`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
