# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `16`
- rows: `251`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 251 | 1.000 | 0.605792 | 0.618724 | -0.012932 | 111 | 140 | 0.709163 | 0.729084 |
| active/recent utility | 251 | 1.000 | 0.605792 | 0.618724 | -0.012932 | 111 | 140 | 0.709163 | 0.729084 |
| strong utility action | 184 | 0.733 | 0.593875 | 0.594435 | -0.000560 | 92 | 92 | 0.684783 | 0.657609 |
| utility damage | 25 | 0.100 | 0.469869 | 0.455431 | 0.014439 | 17 | 8 | 0.400000 | 0.400000 |
| active smoke/inferno | 184 | 0.733 | 0.593875 | 0.594435 | -0.000560 | 92 | 92 | 0.684783 | 0.657609 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 251 | 1.000 | 0.605792 | 0.618724 | -0.012932 | 111 | 140 | 0.709163 | 0.729084 |

## Active Smoke/Inferno Intervals

- `2.5s` - `35.0s`, rows `66`
- `39.5s` - `61.5s`, rows `45`
- `66.5s` - `96.5s`, rows `61`
- `116.0s` - `121.5s`, rows `12`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `116.0`, LSTM `0.7237`, XGBoost `0.8595`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.6805`, XGBoost `0.5504`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `116.5`, LSTM `0.7306`, XGBoost `0.8588`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `117.0`, LSTM `0.7446`, XGBoost `0.8605`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.6765`, XGBoost `0.5618`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.6760`, XGBoost `0.5633`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.8289`, XGBoost `0.7172`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `117.5`, LSTM `0.7527`, XGBoost `0.8632`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.5189`, XGBoost `0.4131`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.5187`, XGBoost `0.4151`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
