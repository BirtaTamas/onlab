# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `16`
- rows: `273`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 273 | 1.000 | 0.426541 | 0.541735 | -0.115193 | 206 | 67 | 0.476190 | 0.424908 |
| active/recent utility | 273 | 1.000 | 0.426541 | 0.541735 | -0.115193 | 206 | 67 | 0.476190 | 0.424908 |
| strong utility action | 168 | 0.615 | 0.506612 | 0.579761 | -0.073148 | 116 | 52 | 0.339286 | 0.339286 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 168 | 0.615 | 0.506612 | 0.579761 | -0.073148 | 116 | 52 | 0.339286 | 0.339286 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 273 | 1.000 | 0.426541 | 0.541735 | -0.115193 | 206 | 67 | 0.476190 | 0.424908 |

## Active Smoke/Inferno Intervals

- `8.0s` - `69.5s`, rows `124`
- `78.5s` - `100.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `95.0`, LSTM `0.0836`, XGBoost `0.4286`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.1474`, XGBoost `0.4872`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.0933`, XGBoost `0.4286`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.1481`, XGBoost `0.4830`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.1504`, XGBoost `0.4844`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.1531`, XGBoost `0.4872`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.1479`, XGBoost `0.4814`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.0996`, XGBoost `0.4323`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.1551`, XGBoost `0.4872`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1025`, XGBoost `0.4323`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
