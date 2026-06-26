# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `3`
- rows: `144`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.029039 | 0.045330 | -0.016291 | 141 | 3 | 1.000000 | 1.000000 |
| active/recent utility | 144 | 1.000 | 0.029039 | 0.045330 | -0.016291 | 141 | 3 | 1.000000 | 1.000000 |
| strong utility action | 53 | 0.368 | 0.056508 | 0.078748 | -0.022239 | 50 | 3 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 53 | 0.368 | 0.056508 | 0.078748 | -0.022239 | 50 | 3 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.029039 | 0.045330 | -0.016291 | 141 | 3 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `32.0s`, rows `53`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.5`, LSTM `0.0935`, XGBoost `0.1722`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0992`, XGBoost `0.1722`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1002`, XGBoost `0.1724`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1036`, XGBoost `0.1724`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1055`, XGBoost `0.1726`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1127`, XGBoost `0.1714`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1167`, XGBoost `0.1723`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1179`, XGBoost `0.1725`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1174`, XGBoost `0.1711`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1273`, XGBoost `0.1723`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
