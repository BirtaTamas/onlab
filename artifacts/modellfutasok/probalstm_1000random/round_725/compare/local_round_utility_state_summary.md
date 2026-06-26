# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `1`
- rows: `118`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 118 | 1.000 | 0.536647 | 0.556088 | -0.019441 | 54 | 64 | 0.576271 | 0.516949 |
| active/recent utility | 118 | 1.000 | 0.536647 | 0.556088 | -0.019441 | 54 | 64 | 0.576271 | 0.516949 |
| strong utility action | 41 | 0.347 | 0.632647 | 0.662407 | -0.029761 | 18 | 23 | 0.804878 | 0.487805 |
| utility damage | 10 | 0.085 | 0.660773 | 0.768879 | -0.108106 | 0 | 10 | 0.800000 | 0.800000 |
| active smoke/inferno | 41 | 0.347 | 0.632647 | 0.662407 | -0.029761 | 18 | 23 | 0.804878 | 0.487805 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 118 | 1.000 | 0.536647 | 0.556088 | -0.019441 | 54 | 64 | 0.576271 | 0.516949 |

## Active Smoke/Inferno Intervals

- `38.5s` - `58.5s`, rows `41`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.5`, LSTM `0.5625`, XGBoost `0.7313`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5874`, XGBoost `0.7358`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.7871`, XGBoost `0.9304`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.8048`, XGBoost `0.9283`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.2441`, XGBoost `0.3623`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5281`, XGBoost `0.6298`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.9014`, XGBoost `0.9843`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.3407`, XGBoost `0.4176`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.9173`, XGBoost `0.9844`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5106`, XGBoost `0.4467`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
