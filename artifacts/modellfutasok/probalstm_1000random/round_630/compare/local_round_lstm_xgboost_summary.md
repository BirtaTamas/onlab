# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `7`
- rows: `293`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.330238 | 0.156220 | 0.459679 | 0.665529 | 0.330238 |
| xgboost | 0.477010 | 0.300170 | 0.763555 | 0.296928 | 0.477010 |

## Closer Per Tick

- lstm: `256`
- xgboost: `37`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
