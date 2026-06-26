# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `12`
- rows: `100`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.228805 | 0.076914 | 0.282865 | 0.940000 | 0.228805 |
| xgboost | 0.327125 | 0.150567 | 0.448829 | 0.640000 | 0.327125 |

## Closer Per Tick

- lstm: `87`
- xgboost: `13`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
