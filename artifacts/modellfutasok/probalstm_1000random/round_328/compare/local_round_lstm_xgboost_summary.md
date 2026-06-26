# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `3`
- rows: `224`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.478252 | 0.251951 | 0.688323 | 0.584821 | 0.521748 |
| xgboost | 0.566101 | 0.356565 | 0.916562 | 0.272321 | 0.433899 |

## Closer Per Tick

- lstm: `166`
- xgboost: `58`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
