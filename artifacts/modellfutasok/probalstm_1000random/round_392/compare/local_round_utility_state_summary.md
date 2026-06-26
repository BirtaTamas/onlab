# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `1`
- rows: `125`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.643780 | 0.662651 | -0.018871 | 55 | 70 | 0.792000 | 0.840000 |
| active/recent utility | 125 | 1.000 | 0.643780 | 0.662651 | -0.018871 | 55 | 70 | 0.792000 | 0.840000 |
| strong utility action | 59 | 0.472 | 0.626207 | 0.641008 | -0.014801 | 35 | 24 | 0.728814 | 0.694915 |
| utility damage | 10 | 0.080 | 0.672686 | 0.844103 | -0.171417 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 51 | 0.408 | 0.622772 | 0.611248 | 0.011524 | 35 | 16 | 0.686275 | 0.647059 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 125 | 1.000 | 0.643780 | 0.662651 | -0.018871 | 55 | 70 | 0.792000 | 0.840000 |

## Active Smoke/Inferno Intervals

- `20.5s` - `45.5s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.5631`, XGBoost `0.7977`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5755`, XGBoost `0.7953`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5811`, XGBoost `0.7937`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5850`, XGBoost `0.7953`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6080`, XGBoost `0.7937`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6911`, XGBoost `0.8524`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6973`, XGBoost `0.8524`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.7449`, XGBoost `0.8986`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.7593`, XGBoost `0.8997`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.7644`, XGBoost `0.9029`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `45.0`, recent_utility `0`
