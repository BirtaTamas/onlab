# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `12`
- rows: `145`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.487027 | 0.251960 | 0.694583 | 0.510345 | 0.487027 |
| xgboost | 0.594333 | 0.377792 | 0.976992 | 0.344828 | 0.594333 |

## Closer Per Tick

- lstm: `140`
- xgboost: `5`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
