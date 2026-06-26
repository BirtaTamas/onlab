# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `5`
- rows: `191`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.354849 | 0.447338 | -0.092489 | 191 | 0 | 0.963351 | 0.403141 |
| active/recent utility | 191 | 1.000 | 0.354849 | 0.447338 | -0.092489 | 191 | 0 | 0.963351 | 0.403141 |
| strong utility action | 178 | 0.932 | 0.371063 | 0.468473 | -0.097410 | 178 | 0 | 0.960674 | 0.359551 |
| utility damage | 20 | 0.105 | 0.450891 | 0.520904 | -0.070013 | 20 | 0 | 1.000000 | 0.100000 |
| active smoke/inferno | 168 | 0.880 | 0.366841 | 0.467938 | -0.101096 | 168 | 0 | 0.958333 | 0.321429 |
| recent utility last 5s | 10 | 0.052 | 0.441980 | 0.477464 | -0.035484 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 191 | 1.000 | 0.354849 | 0.447338 | -0.092489 | 191 | 0 | 0.963351 | 0.403141 |

## Active Smoke/Inferno Intervals

- `7.0s` - `90.5s`, rows `168`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.0906`, XGBoost `0.3367`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.1000`, XGBoost `0.3381`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0998`, XGBoost `0.3373`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.1027`, XGBoost `0.3367`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.1028`, XGBoost `0.3367`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1038`, XGBoost `0.3373`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.1055`, XGBoost `0.3381`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.1061`, XGBoost `0.3367`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.1077`, XGBoost `0.3367`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.1090`, XGBoost `0.3371`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
