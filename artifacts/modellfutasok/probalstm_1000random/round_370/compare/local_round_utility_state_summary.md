# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `13`
- rows: `144`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.616747 | 0.639777 | -0.023031 | 26 | 118 | 0.770833 | 0.777778 |
| active/recent utility | 144 | 1.000 | 0.616747 | 0.639777 | -0.023031 | 26 | 118 | 0.770833 | 0.777778 |
| strong utility action | 100 | 0.694 | 0.667027 | 0.697005 | -0.029978 | 13 | 87 | 0.760000 | 0.770000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 100 | 0.694 | 0.667027 | 0.697005 | -0.029978 | 13 | 87 | 0.760000 | 0.770000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.616747 | 0.639777 | -0.023031 | 26 | 118 | 0.770833 | 0.777778 |

## Active Smoke/Inferno Intervals

- `9.5s` - `31.0s`, rows `44`
- `44.0s` - `71.5s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.3367`, XGBoost `0.4704`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.8437`, XGBoost `0.9554`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.3213`, XGBoost `0.4313`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.8320`, XGBoost `0.9417`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.8460`, XGBoost `0.9554`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.8466`, XGBoost `0.9556`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.8541`, XGBoost `0.9552`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5849`, XGBoost `0.6848`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.3395`, XGBoost `0.4307`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.8722`, XGBoost `0.9556`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
