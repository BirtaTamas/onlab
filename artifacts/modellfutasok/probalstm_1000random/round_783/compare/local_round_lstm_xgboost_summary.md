# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `17`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.211358 | 0.082198 | 0.274792 | 0.843478 | 0.788642 |
| xgboost | 0.226990 | 0.072215 | 0.278152 | 0.921739 | 0.773010 |

## Closer Per Tick

- lstm: `124`
- xgboost: `106`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
