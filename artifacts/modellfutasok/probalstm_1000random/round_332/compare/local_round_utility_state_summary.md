# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `12`
- rows: `125`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.759924 | 0.734503 | 0.025421 | 66 | 59 | 1.000000 | 0.800000 |
| active/recent utility | 125 | 1.000 | 0.759924 | 0.734503 | 0.025421 | 66 | 59 | 1.000000 | 0.800000 |
| strong utility action | 113 | 0.904 | 0.775664 | 0.757809 | 0.017855 | 54 | 59 | 1.000000 | 0.778761 |
| utility damage | 12 | 0.096 | 0.873820 | 0.910425 | -0.036605 | 1 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 113 | 0.904 | 0.775664 | 0.757809 | 0.017855 | 54 | 59 | 1.000000 | 0.778761 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 125 | 1.000 | 0.759924 | 0.734503 | 0.025421 | 66 | 59 | 1.000000 | 0.800000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `62.0s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.0`, LSTM `0.6249`, XGBoost `0.4905`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6130`, XGBoost `0.4881`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6143`, XGBoost `0.4923`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.6128`, XGBoost `0.4910`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.6088`, XGBoost `0.4889`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6095`, XGBoost `0.4908`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6106`, XGBoost `0.4928`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.6105`, XGBoost `0.4939`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.6040`, XGBoost `0.4904`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6103`, XGBoost `0.4967`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
