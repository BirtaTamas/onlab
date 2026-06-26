# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `2`
- rows: `198`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.936775 | 0.975165 | -0.038390 | 0 | 198 | 1.000000 | 1.000000 |
| active/recent utility | 198 | 1.000 | 0.936775 | 0.975165 | -0.038390 | 0 | 198 | 1.000000 | 1.000000 |
| strong utility action | 129 | 0.652 | 0.928374 | 0.970041 | -0.041667 | 0 | 129 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.051 | 0.761989 | 0.900066 | -0.138077 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 127 | 0.641 | 0.930290 | 0.971118 | -0.040828 | 0 | 127 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 198 | 1.000 | 0.936775 | 0.975165 | -0.038390 | 0 | 198 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `40.0s`, rows `60`
- `48.0s` - `81.0s`, rows `67`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.6844`, XGBoost `0.9035`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.6814`, XGBoost `0.8876`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.7001`, XGBoost `0.8849`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.7077`, XGBoost `0.8870`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.7216`, XGBoost `0.8977`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.7373`, XGBoost `0.8914`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.7552`, XGBoost `0.8914`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.7556`, XGBoost `0.8914`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.7857`, XGBoost `0.9097`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.7837`, XGBoost `0.9074`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `19.0`, recent_utility `0`
