# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `14`
- rows: `188`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.225330 | 0.114267 | 0.315074 | 0.579787 | 0.225330 |
| xgboost | 0.223090 | 0.109126 | 0.307633 | 0.765957 | 0.223090 |

## Closer Per Tick

- lstm: `134`
- xgboost: `54`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
