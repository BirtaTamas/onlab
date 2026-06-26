# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `16`
- rows: `184`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.470287 | 0.253899 | 0.689560 | 0.396739 | 0.470287 |
| xgboost | 0.509994 | 0.285701 | 0.766218 | 0.375000 | 0.509994 |

## Closer Per Tick

- lstm: `133`
- xgboost: `51`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
