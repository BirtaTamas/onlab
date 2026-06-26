# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `7`
- rows: `133`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.772494 | 0.807453 | -0.034959 | 12 | 121 | 0.887218 | 0.894737 |
| active/recent utility | 133 | 1.000 | 0.772494 | 0.807453 | -0.034959 | 12 | 121 | 0.887218 | 0.894737 |
| strong utility action | 126 | 0.947 | 0.767106 | 0.803482 | -0.036376 | 11 | 115 | 0.880952 | 0.888889 |
| utility damage | 21 | 0.158 | 0.714262 | 0.761207 | -0.046945 | 2 | 19 | 1.000000 | 1.000000 |
| active smoke/inferno | 116 | 0.872 | 0.772265 | 0.811807 | -0.039542 | 6 | 110 | 0.870690 | 0.879310 |
| recent utility last 5s | 10 | 0.075 | 0.707267 | 0.706912 | 0.000355 | 5 | 5 | 1.000000 | 1.000000 |
| flash effect present | 133 | 1.000 | 0.772494 | 0.807453 | -0.034959 | 12 | 121 | 0.887218 | 0.894737 |

## Active Smoke/Inferno Intervals

- `6.5s` - `64.0s`, rows `116`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.0`, LSTM `0.6824`, XGBoost `0.8149`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7006`, XGBoost `0.8152`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.8385`, XGBoost `0.9359`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.8393`, XGBoost `0.9359`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.8398`, XGBoost `0.9359`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5965`, XGBoost `0.6907`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.7208`, XGBoost `0.8148`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7227`, XGBoost `0.8144`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5445`, XGBoost `0.6359`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7239`, XGBoost `0.8148`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `6.0`, recent_utility `0`
