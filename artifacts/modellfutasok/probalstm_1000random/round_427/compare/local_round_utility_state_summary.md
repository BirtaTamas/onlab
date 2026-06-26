# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `17`
- rows: `244`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 244 | 1.000 | 0.630979 | 0.690185 | -0.059206 | 179 | 65 | 0.032787 | 0.241803 |
| active/recent utility | 244 | 1.000 | 0.630979 | 0.690185 | -0.059206 | 179 | 65 | 0.032787 | 0.241803 |
| strong utility action | 226 | 0.926 | 0.641101 | 0.705420 | -0.064319 | 170 | 56 | 0.017699 | 0.207965 |
| utility damage | 10 | 0.041 | 0.621916 | 0.609097 | 0.012818 | 4 | 6 | 0.000000 | 0.100000 |
| active smoke/inferno | 226 | 0.926 | 0.641101 | 0.705420 | -0.064319 | 170 | 56 | 0.017699 | 0.207965 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 244 | 1.000 | 0.630979 | 0.690185 | -0.059206 | 179 | 65 | 0.032787 | 0.241803 |

## Active Smoke/Inferno Intervals

- `9.0s` - `121.5s`, rows `226`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `94.0`, LSTM `0.5882`, XGBoost `0.8576`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.5808`, XGBoost `0.8457`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.5994`, XGBoost `0.8630`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.5983`, XGBoost `0.8616`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.6002`, XGBoost `0.8583`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.6003`, XGBoost `0.8583`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.5893`, XGBoost `0.8457`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.6044`, XGBoost `0.8576`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.6059`, XGBoost `0.8583`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.5933`, XGBoost `0.8457`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
