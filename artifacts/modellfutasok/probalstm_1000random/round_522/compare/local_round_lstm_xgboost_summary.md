# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `8`
- rows: `245`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.016418 | 0.000452 | 0.016648 | 1.000000 | 0.016418 |
| xgboost | 0.045757 | 0.002584 | 0.047103 | 1.000000 | 0.045757 |

## Closer Per Tick

- lstm: `231`
- xgboost: `14`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
