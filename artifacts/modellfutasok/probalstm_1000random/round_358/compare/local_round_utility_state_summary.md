# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `9`
- rows: `234`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 234 | 1.000 | 0.090386 | 0.141417 | -0.051031 | 218 | 16 | 1.000000 | 1.000000 |
| active/recent utility | 234 | 1.000 | 0.090386 | 0.141417 | -0.051031 | 218 | 16 | 1.000000 | 1.000000 |
| strong utility action | 121 | 0.517 | 0.140794 | 0.197281 | -0.056488 | 106 | 15 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.043 | 0.302907 | 0.300478 | 0.002429 | 5 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 121 | 0.517 | 0.140794 | 0.197281 | -0.056488 | 106 | 15 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 234 | 1.000 | 0.090386 | 0.141417 | -0.051031 | 218 | 16 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `44.0s`, rows `74`
- `53.0s` - `76.0s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.5`, LSTM `0.0940`, XGBoost `0.3026`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1081`, XGBoost `0.3044`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1074`, XGBoost `0.3011`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1085`, XGBoost `0.2958`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1123`, XGBoost `0.2992`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1206`, XGBoost `0.2989`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1212`, XGBoost `0.2993`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1203`, XGBoost `0.2930`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1296`, XGBoost `0.3008`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.0388`, XGBoost `0.2058`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
