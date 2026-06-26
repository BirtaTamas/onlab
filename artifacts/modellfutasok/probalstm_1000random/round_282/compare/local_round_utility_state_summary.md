# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `4`
- rows: `202`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 202 | 1.000 | 0.687390 | 0.666901 | 0.020489 | 126 | 76 | 1.000000 | 0.970297 |
| active/recent utility | 202 | 1.000 | 0.687390 | 0.666901 | 0.020489 | 126 | 76 | 1.000000 | 0.970297 |
| strong utility action | 159 | 0.787 | 0.701228 | 0.682768 | 0.018460 | 97 | 62 | 1.000000 | 0.993711 |
| utility damage | 34 | 0.168 | 0.705521 | 0.686689 | 0.018832 | 20 | 14 | 1.000000 | 1.000000 |
| active smoke/inferno | 147 | 0.728 | 0.705398 | 0.685517 | 0.019882 | 90 | 57 | 1.000000 | 0.993197 |
| recent utility last 5s | 10 | 0.050 | 0.635914 | 0.643550 | -0.007636 | 4 | 6 | 1.000000 | 1.000000 |
| flash effect present | 202 | 1.000 | 0.687390 | 0.666901 | 0.020489 | 126 | 76 | 1.000000 | 0.970297 |

## Active Smoke/Inferno Intervals

- `8.5s` - `50.5s`, rows `85`
- `62.0s` - `85.5s`, rows `48`
- `92.5s` - `99.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `8.5`, LSTM `0.6296`, XGBoost `0.4996`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6278`, XGBoost `0.5027`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.6221`, XGBoost `0.5036`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6198`, XGBoost `0.5049`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.6144`, XGBoost `0.5049`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6084`, XGBoost `0.5020`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `8.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6038`, XGBoost `0.5034`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `8.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.7314`, XGBoost `0.6354`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7380`, XGBoost `0.6422`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.7356`, XGBoost `0.6422`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
