# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `17`
- rows: `267`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.404221 | 0.199747 | 0.564224 | 0.483146 | 0.595779 |
| xgboost | 0.361202 | 0.161792 | 0.483114 | 0.868914 | 0.638798 |

## Closer Per Tick

- lstm: `36`
- xgboost: `231`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
