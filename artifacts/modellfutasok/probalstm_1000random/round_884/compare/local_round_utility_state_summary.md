# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `10`
- rows: `192`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 192 | 1.000 | 0.317080 | 0.468325 | -0.151245 | 168 | 24 | 0.713542 | 0.489583 |
| active/recent utility | 192 | 1.000 | 0.317080 | 0.468325 | -0.151245 | 168 | 24 | 0.713542 | 0.489583 |
| strong utility action | 155 | 0.807 | 0.332017 | 0.480756 | -0.148739 | 131 | 24 | 0.645161 | 0.477419 |
| utility damage | 32 | 0.167 | 0.242519 | 0.451990 | -0.209472 | 32 | 0 | 0.718750 | 0.687500 |
| active smoke/inferno | 153 | 0.797 | 0.334571 | 0.482161 | -0.147590 | 129 | 24 | 0.640523 | 0.470588 |
| recent utility last 5s | 10 | 0.052 | 0.049660 | 0.323160 | -0.273500 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 192 | 1.000 | 0.317080 | 0.468325 | -0.151245 | 168 | 24 | 0.713542 | 0.489583 |

## Active Smoke/Inferno Intervals

- `8.5s` - `57.0s`, rows `98`
- `61.0s` - `66.0s`, rows `11`
- `71.0s` - `92.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.5`, LSTM `0.0275`, XGBoost `0.4375`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.0280`, XGBoost `0.4146`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0846`, XGBoost `0.4426`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `5.0`, recent_utility `1`
- seconds `71.5`, LSTM `0.0438`, XGBoost `0.3744`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0468`, XGBoost `0.3744`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0487`, XGBoost `0.3744`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0553`, XGBoost `0.3744`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0568`, XGBoost `0.3744`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0558`, XGBoost `0.3731`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.0588`, XGBoost `0.3731`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
