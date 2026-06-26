# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mibr-bo3-vjmAHfXA4PQfROTmirSCCF/vitality-vs-mibr-m2-inferno.csv`
- round_num: `7`
- rows: `212`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.196278 | 0.055381 | 0.234281 | 0.957547 | 0.196278 |
| xgboost | 0.291268 | 0.096577 | 0.357373 | 0.943396 | 0.291268 |

## Closer Per Tick

- lstm: `206`
- xgboost: `6`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
