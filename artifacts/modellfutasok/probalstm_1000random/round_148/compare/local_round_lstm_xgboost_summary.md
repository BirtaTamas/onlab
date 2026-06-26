# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `16`
- rows: `273`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.426541 | 0.264709 | 0.698316 | 0.476190 | 0.426541 |
| xgboost | 0.541735 | 0.327502 | 0.864035 | 0.424908 | 0.541735 |

## Closer Per Tick

- lstm: `206`
- xgboost: `67`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
