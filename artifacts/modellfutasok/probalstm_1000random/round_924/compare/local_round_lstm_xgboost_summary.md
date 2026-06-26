# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-fnatic-vs-legacy-bo3-XoJZ8zL16kSaGnHRZrLL4s/legacy-vs-fnatic-m1-ancient.csv`
- round_num: `4`
- rows: `137`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.335733 | 0.122402 | 0.419993 | 1.000000 | 0.664267 |
| xgboost | 0.287065 | 0.087857 | 0.343647 | 1.000000 | 0.712935 |

## Closer Per Tick

- lstm: `51`
- xgboost: `86`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
