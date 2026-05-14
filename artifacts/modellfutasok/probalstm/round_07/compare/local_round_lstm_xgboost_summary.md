# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\blast_austin_major_stage_1\blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX\flyquest-vs-fluxo-ancient.csv`
- round_num: `8`
- rows: `231`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.404426 | 0.218982 | 0.630482 | 0.779221 | 0.595574 |
| xgboost | 0.394552 | 0.210559 | 0.588797 | 0.787879 | 0.605448 |

## Closer Per Tick

- lstm: `108`
- xgboost: `123`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
