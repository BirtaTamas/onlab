# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `4`
- rows: `187`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.245636 | 0.129449 | 0.350536 | 0.598930 | 0.245636 |
| xgboost | 0.245618 | 0.126079 | 0.348385 | 0.743316 | 0.245618 |

## Closer Per Tick

- lstm: `140`
- xgboost: `47`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
