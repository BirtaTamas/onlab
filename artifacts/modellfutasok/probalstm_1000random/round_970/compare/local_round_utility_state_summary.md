# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `13`
- rows: `138`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.114132 | 0.164422 | -0.050291 | 127 | 11 | 0.884058 | 0.782609 |
| active/recent utility | 138 | 1.000 | 0.114132 | 0.164422 | -0.050291 | 127 | 11 | 0.884058 | 0.782609 |
| strong utility action | 44 | 0.319 | 0.009874 | 0.049889 | -0.040015 | 44 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.319 | 0.009874 | 0.049889 | -0.040015 | 44 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 138 | 1.000 | 0.114132 | 0.164422 | -0.050291 | 127 | 11 | 0.884058 | 0.782609 |

## Active Smoke/Inferno Intervals

- `21.5s` - `43.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.0`, LSTM `0.0163`, XGBoost `0.2047`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.0144`, XGBoost `0.1995`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.0261`, XGBoost `0.1963`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0292`, XGBoost `0.1963`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0232`, XGBoost `0.0921`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0228`, XGBoost `0.0897`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0243`, XGBoost `0.0859`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0300`, XGBoost `0.0859`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0303`, XGBoost `0.0851`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0392`, XGBoost `0.0877`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
