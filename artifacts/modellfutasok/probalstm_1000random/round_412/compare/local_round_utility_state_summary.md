# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-pain-bo3-zcuZjSa9VUSMkJoK5k8I3c/gamerlegion-vs-pain-m3-mirage.csv`
- round_num: `1`
- rows: `142`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.158997 | 0.186715 | -0.027719 | 127 | 15 | 0.852113 | 0.725352 |
| active/recent utility | 142 | 1.000 | 0.158997 | 0.186715 | -0.027719 | 127 | 15 | 0.852113 | 0.725352 |
| strong utility action | 53 | 0.373 | 0.073774 | 0.129720 | -0.055946 | 47 | 6 | 1.000000 | 0.962264 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 53 | 0.373 | 0.073774 | 0.129720 | -0.055946 | 47 | 6 | 1.000000 | 0.962264 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.158997 | 0.186715 | -0.027719 | 127 | 15 | 0.852113 | 0.725352 |

## Active Smoke/Inferno Intervals

- `18.5s` - `44.5s`, rows `53`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.2013`, XGBoost `0.4410`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2008`, XGBoost `0.4396`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.2179`, XGBoost `0.4478`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.2113`, XGBoost `0.4384`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.2436`, XGBoost `0.4358`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0248`, XGBoost `0.1777`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0246`, XGBoost `0.1752`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0273`, XGBoost `0.1777`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0284`, XGBoost `0.1777`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0287`, XGBoost `0.1777`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
