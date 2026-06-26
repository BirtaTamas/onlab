# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `16`
- rows: `100`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 100 | 1.000 | 0.015063 | 0.037774 | -0.022711 | 99 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 100 | 1.000 | 0.015063 | 0.037774 | -0.022711 | 99 | 1 | 1.000000 | 1.000000 |
| strong utility action | 44 | 0.440 | 0.025720 | 0.061593 | -0.035872 | 44 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.440 | 0.025720 | 0.061593 | -0.035872 | 44 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.100 | 0.016232 | 0.058986 | -0.042754 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 100 | 1.000 | 0.015063 | 0.037774 | -0.022711 | 99 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `29.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.0`, LSTM `0.0108`, XGBoost `0.0579`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0073`, XGBoost `0.0517`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0129`, XGBoost `0.0571`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `1`
- seconds `16.5`, LSTM `0.0181`, XGBoost `0.0619`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0164`, XGBoost `0.0598`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `1`
- seconds `16.0`, LSTM `0.0179`, XGBoost `0.0613`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `14.5`, LSTM `0.0168`, XGBoost `0.0598`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `1`
- seconds `13.0`, LSTM `0.0159`, XGBoost `0.0588`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `1`
- seconds `15.0`, LSTM `0.0184`, XGBoost `0.0612`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `6.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.0090`, XGBoost `0.0517`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
