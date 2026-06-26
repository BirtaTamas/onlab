# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `7`
- rows: `140`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.343520 | 0.132859 | 0.439123 | 0.821429 | 0.656480 |
| xgboost | 0.331947 | 0.147475 | 0.446324 | 0.771429 | 0.668053 |

## Closer Per Tick

- lstm: `77`
- xgboost: `63`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
