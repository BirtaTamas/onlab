# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `3`
- rows: `179`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 179 | 1.000 | 0.044965 | 0.065321 | -0.020356 | 168 | 11 | 1.000000 | 1.000000 |
| active/recent utility | 179 | 1.000 | 0.044965 | 0.065321 | -0.020356 | 168 | 11 | 1.000000 | 1.000000 |
| strong utility action | 155 | 0.866 | 0.048117 | 0.065907 | -0.017790 | 144 | 11 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 145 | 0.810 | 0.049329 | 0.065844 | -0.016515 | 134 | 11 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.056 | 0.030534 | 0.066817 | -0.036283 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 179 | 1.000 | 0.044965 | 0.065321 | -0.020356 | 168 | 11 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `72.0s`, rows `131`
- `82.5s` - `89.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.0610`, XGBoost `0.1197`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.0694`, XGBoost `0.1275`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0231`, XGBoost `0.0713`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0211`, XGBoost `0.0684`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0260`, XGBoost `0.0720`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0258`, XGBoost `0.0712`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0258`, XGBoost `0.0708`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0272`, XGBoost `0.0719`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0246`, XGBoost `0.0691`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.0225`, XGBoost `0.0668`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
