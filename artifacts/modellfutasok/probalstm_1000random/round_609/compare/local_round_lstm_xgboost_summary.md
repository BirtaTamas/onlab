# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `3`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.029039 | 0.002731 | 0.030520 | 1.000000 | 0.029039 |
| xgboost | 0.045330 | 0.006291 | 0.048868 | 1.000000 | 0.045330 |

## Closer Per Tick

- lstm: `141`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
