# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `5`
- rows: `162`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 162 | 1.000 | 0.008677 | 0.040943 | -0.032266 | 137 | 25 | 1.000000 | 1.000000 |
| active/recent utility | 162 | 1.000 | 0.008677 | 0.040943 | -0.032266 | 137 | 25 | 1.000000 | 1.000000 |
| strong utility action | 80 | 0.494 | 0.007880 | 0.045716 | -0.037836 | 75 | 5 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 70 | 0.432 | 0.006142 | 0.042619 | -0.036476 | 65 | 5 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.062 | 0.020041 | 0.067395 | -0.047354 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 162 | 1.000 | 0.008677 | 0.040943 | -0.032266 | 137 | 25 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `25.5s` - `32.0s`, rows `14`
- `34.0s` - `61.5s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.0049`, XGBoost `0.0675`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0053`, XGBoost `0.0675`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0042`, XGBoost `0.0650`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0047`, XGBoost `0.0650`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0050`, XGBoost `0.0650`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.0075`, XGBoost `0.0675`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0052`, XGBoost `0.0650`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0079`, XGBoost `0.0675`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0081`, XGBoost `0.0675`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0083`, XGBoost `0.0675`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
