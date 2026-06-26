# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `7`
- rows: `275`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.372155 | 0.196284 | 0.538411 | 0.574545 | 0.372155 |
| xgboost | 0.392426 | 0.208018 | 0.567884 | 0.567273 | 0.392426 |

## Closer Per Tick

- lstm: `176`
- xgboost: `99`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
