# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `1`
- rows: `130`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.108186 | 0.049740 | 0.146657 | 0.838462 | 0.108186 |
| xgboost | 0.113349 | 0.047610 | 0.149230 | 1.000000 | 0.113349 |

## Closer Per Tick

- lstm: `100`
- xgboost: `30`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
