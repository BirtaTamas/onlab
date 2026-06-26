# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-liquid-vs-mouz-bo3-heKnTsZGq8rrQ4y9Yn2KrI/liquid-vs-mouz-m2-train.csv`
- round_num: `15`
- rows: `130`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.192767 | 0.065902 | 0.237729 | 0.992308 | 0.807233 |
| xgboost | 0.234083 | 0.102066 | 0.309755 | 0.976923 | 0.765917 |

## Closer Per Tick

- lstm: `73`
- xgboost: `57`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
