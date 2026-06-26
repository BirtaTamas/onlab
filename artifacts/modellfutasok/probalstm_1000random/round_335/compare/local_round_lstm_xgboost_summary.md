# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `4`
- rows: `266`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.512094 | 0.354004 | 0.927528 | 0.300752 | 0.512094 |
| xgboost | 0.595475 | 0.459084 | 1.309071 | 0.236842 | 0.595475 |

## Closer Per Tick

- lstm: `266`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
