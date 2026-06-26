# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `37`
- rows: `142`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.431559 | 0.267451 | 0.700572 | 0.436620 | 0.431559 |
| xgboost | 0.456896 | 0.282589 | 0.744709 | 0.422535 | 0.456896 |

## Closer Per Tick

- lstm: `114`
- xgboost: `28`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
