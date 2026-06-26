# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `4`
- rows: `181`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.426446 | 0.191911 | 0.568415 | 0.983425 | 0.573554 |
| xgboost | 0.434631 | 0.204458 | 0.589353 | 0.839779 | 0.565369 |

## Closer Per Tick

- lstm: `118`
- xgboost: `63`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
