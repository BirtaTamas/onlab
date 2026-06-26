# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `13`
- rows: `215`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.263951 | 0.362179 | -0.098227 | 213 | 2 | 0.883721 | 0.497674 |
| active/recent utility | 215 | 1.000 | 0.263951 | 0.362179 | -0.098227 | 213 | 2 | 0.883721 | 0.497674 |
| strong utility action | 46 | 0.214 | 0.016383 | 0.127134 | -0.110752 | 46 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 46 | 0.214 | 0.016383 | 0.127134 | -0.110752 | 46 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 215 | 1.000 | 0.263951 | 0.362179 | -0.098227 | 213 | 2 | 0.883721 | 0.497674 |

## Active Smoke/Inferno Intervals

- `77.5s` - `100.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.0`, LSTM `0.0308`, XGBoost `0.2915`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0356`, XGBoost `0.2915`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.0339`, XGBoost `0.2892`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.0361`, XGBoost `0.2907`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.0404`, XGBoost `0.2948`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.0355`, XGBoost `0.2881`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.0366`, XGBoost `0.2881`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0447`, XGBoost `0.2956`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.0385`, XGBoost `0.2892`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.0447`, XGBoost `0.2952`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
