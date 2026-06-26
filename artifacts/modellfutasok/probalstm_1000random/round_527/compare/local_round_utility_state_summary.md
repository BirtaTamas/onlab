# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `13`
- rows: `234`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 234 | 1.000 | 0.391934 | 0.502900 | -0.110967 | 230 | 4 | 0.512821 | 0.358974 |
| active/recent utility | 234 | 1.000 | 0.391934 | 0.502900 | -0.110967 | 230 | 4 | 0.512821 | 0.358974 |
| strong utility action | 76 | 0.325 | 0.507695 | 0.626528 | -0.118833 | 74 | 2 | 0.210526 | 0.197368 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 76 | 0.325 | 0.507695 | 0.626528 | -0.118833 | 74 | 2 | 0.210526 | 0.197368 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 234 | 1.000 | 0.391934 | 0.502900 | -0.110967 | 230 | 4 | 0.512821 | 0.358974 |

## Active Smoke/Inferno Intervals

- `25.5s` - `63.0s`, rows `76`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.6101`, XGBoost `0.8835`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6259`, XGBoost `0.8845`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.5183`, XGBoost `0.7637`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6412`, XGBoost `0.8845`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.4246`, XGBoost `0.6668`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6425`, XGBoost `0.8845`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.6256`, XGBoost `0.8666`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1618`, XGBoost `0.3963`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6308`, XGBoost `0.8591`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.4464`, XGBoost `0.6668`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
