# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `2`
- rows: `268`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.062393 | 0.013135 | 0.070654 | 1.000000 | 0.062393 |
| xgboost | 0.073040 | 0.020137 | 0.086317 | 1.000000 | 0.073040 |

## Closer Per Tick

- lstm: `158`
- xgboost: `110`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
