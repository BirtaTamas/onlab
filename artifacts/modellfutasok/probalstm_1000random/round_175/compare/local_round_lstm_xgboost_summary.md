# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `22`
- rows: `238`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.346628 | 0.152841 | 0.464849 | 0.621849 | 0.653372 |
| xgboost | 0.320278 | 0.132144 | 0.419153 | 0.936975 | 0.679722 |

## Closer Per Tick

- lstm: `63`
- xgboost: `175`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
