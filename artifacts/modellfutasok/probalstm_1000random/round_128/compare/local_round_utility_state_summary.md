# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-b8-bo3-rUWlZLFFckLiQv1C1wSlHb/g2-vs-b8-m3-ancient.csv`
- round_num: `7`
- rows: `291`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 291 | 1.000 | 0.443960 | 0.411878 | 0.032082 | 94 | 197 | 0.292096 | 0.422680 |
| active/recent utility | 291 | 1.000 | 0.443960 | 0.411878 | 0.032082 | 94 | 197 | 0.292096 | 0.422680 |
| strong utility action | 190 | 0.653 | 0.478496 | 0.444354 | 0.034143 | 49 | 141 | 0.215789 | 0.415789 |
| utility damage | 25 | 0.086 | 0.644628 | 0.569950 | 0.074678 | 0 | 25 | 0.000000 | 0.000000 |
| active smoke/inferno | 190 | 0.653 | 0.478496 | 0.444354 | 0.034143 | 49 | 141 | 0.215789 | 0.415789 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 291 | 1.000 | 0.443960 | 0.411878 | 0.032082 | 94 | 197 | 0.292096 | 0.422680 |

## Active Smoke/Inferno Intervals

- `6.5s` - `44.5s`, rows `77`
- `65.0s` - `70.0s`, rows `11`
- `71.0s` - `99.5s`, rows `58`
- `104.0s` - `125.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `108.5`, LSTM `0.0781`, XGBoost `0.2155`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.5003`, XGBoost `0.6342`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.0861`, XGBoost `0.2126`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.6822`, XGBoost `0.5558`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.5103`, XGBoost `0.6349`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.5095`, XGBoost `0.6328`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.1103`, XGBoost `0.2222`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.6863`, XGBoost `0.5790`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6576`, XGBoost `0.5577`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.6753`, XGBoost `0.5790`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
