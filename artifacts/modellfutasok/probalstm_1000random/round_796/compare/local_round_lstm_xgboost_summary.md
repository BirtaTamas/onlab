# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `8`
- rows: `133`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.242121 | 0.065458 | 0.282758 | 1.000000 | 0.757879 |
| xgboost | 0.223002 | 0.057643 | 0.258510 | 1.000000 | 0.776998 |

## Closer Per Tick

- lstm: `29`
- xgboost: `104`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
