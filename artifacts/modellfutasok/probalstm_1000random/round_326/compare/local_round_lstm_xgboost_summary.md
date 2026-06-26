# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `16`
- rows: `141`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.007355 | 0.000152 | 0.007433 | 1.000000 | 0.007355 |
| xgboost | 0.019891 | 0.000787 | 0.020296 | 1.000000 | 0.019891 |

## Closer Per Tick

- lstm: `98`
- xgboost: `43`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
