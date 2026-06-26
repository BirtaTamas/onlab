# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `27`
- rows: `223`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 223 | 1.000 | 0.554619 | 0.605780 | -0.051160 | 108 | 115 | 0.690583 | 0.762332 |
| active/recent utility | 223 | 1.000 | 0.554619 | 0.605780 | -0.051160 | 108 | 115 | 0.690583 | 0.762332 |
| strong utility action | 142 | 0.637 | 0.753258 | 0.733618 | 0.019640 | 97 | 45 | 0.971831 | 0.964789 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 141 | 0.632 | 0.756744 | 0.733974 | 0.022770 | 97 | 44 | 0.978723 | 0.964539 |
| recent utility last 5s | 1 | 0.004 | 0.261725 | 0.683373 | -0.421647 | 0 | 1 | 0.000000 | 1.000000 |
| flash effect present | 223 | 1.000 | 0.554619 | 0.605780 | -0.051160 | 108 | 115 | 0.690583 | 0.762332 |

## Active Smoke/Inferno Intervals

- `8.0s` - `78.0s`, rows `141`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `111.0`, LSTM `0.2617`, XGBoost `0.6834`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `77.5`, LSTM `0.3771`, XGBoost `0.4935`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.8317`, XGBoost `0.7333`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.3963`, XGBoost `0.4935`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.8933`, XGBoost `0.8043`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.8938`, XGBoost `0.8078`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.8924`, XGBoost `0.8089`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.8918`, XGBoost `0.8089`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.8915`, XGBoost `0.8089`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.8806`, XGBoost `0.7983`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
