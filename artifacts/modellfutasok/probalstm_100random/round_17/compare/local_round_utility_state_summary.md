# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `6`
- rows: `253`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 253 | 1.000 | 0.611337 | 0.716034 | -0.104697 | 65 | 188 | 0.695652 | 0.897233 |
| active/recent utility | 253 | 1.000 | 0.611337 | 0.716034 | -0.104697 | 65 | 188 | 0.695652 | 0.897233 |
| strong utility action | 124 | 0.490 | 0.747101 | 0.797846 | -0.050745 | 51 | 73 | 1.000000 | 1.000000 |
| utility damage | 28 | 0.111 | 0.895181 | 0.876111 | 0.019069 | 15 | 13 | 1.000000 | 1.000000 |
| active smoke/inferno | 124 | 0.490 | 0.747101 | 0.797846 | -0.050745 | 51 | 73 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 253 | 1.000 | 0.611337 | 0.716034 | -0.104697 | 65 | 188 | 0.695652 | 0.897233 |

## Active Smoke/Inferno Intervals

- `7.0s` - `46.5s`, rows `80`
- `52.5s` - `74.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.0`, LSTM `0.5030`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.5044`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5092`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5103`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.5191`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5284`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.5297`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5303`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.5300`, XGBoost `0.7550`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.5316`, XGBoost `0.7553`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
