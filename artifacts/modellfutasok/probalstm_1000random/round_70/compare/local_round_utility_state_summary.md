# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `3`
- rows: `146`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.237771 | 0.421777 | -0.184006 | 146 | 0 | 0.780822 | 0.636986 |
| active/recent utility | 146 | 1.000 | 0.237771 | 0.421777 | -0.184006 | 146 | 0 | 0.780822 | 0.636986 |
| strong utility action | 106 | 0.726 | 0.261802 | 0.438723 | -0.176922 | 106 | 0 | 0.698113 | 0.613208 |
| utility damage | 1 | 0.007 | 0.050010 | 0.317697 | -0.267687 | 1 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 105 | 0.719 | 0.263819 | 0.439876 | -0.176057 | 105 | 0 | 0.695238 | 0.609524 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 146 | 1.000 | 0.237771 | 0.421777 | -0.184006 | 146 | 0 | 0.780822 | 0.636986 |

## Active Smoke/Inferno Intervals

- `6.0s` - `58.0s`, rows `105`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.5`, LSTM `0.0534`, XGBoost `0.4768`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0566`, XGBoost `0.4790`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.1120`, XGBoost `0.4802`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1193`, XGBoost `0.4826`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1194`, XGBoost `0.4788`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1296`, XGBoost `0.4835`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1643`, XGBoost `0.4868`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.1010`, XGBoost `0.4100`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.1120`, XGBoost `0.4109`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.1187`, XGBoost `0.3997`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
