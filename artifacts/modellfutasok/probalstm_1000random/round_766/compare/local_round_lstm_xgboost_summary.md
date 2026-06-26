# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `15`
- rows: `164`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.429396 | 0.187315 | 0.565806 | 0.957317 | 0.570604 |
| xgboost | 0.563723 | 0.322224 | 0.840641 | 0.030488 | 0.436277 |

## Closer Per Tick

- lstm: `161`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
