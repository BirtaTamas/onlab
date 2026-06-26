# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `4`
- rows: `175`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 175 | 1.000 | 0.228374 | 0.270049 | -0.041675 | 144 | 31 | 0.977143 | 0.994286 |
| active/recent utility | 175 | 1.000 | 0.228374 | 0.270049 | -0.041675 | 144 | 31 | 0.977143 | 0.994286 |
| strong utility action | 154 | 0.880 | 0.213056 | 0.258854 | -0.045798 | 125 | 29 | 0.974026 | 0.993506 |
| utility damage | 10 | 0.057 | 0.319819 | 0.463626 | -0.143807 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 154 | 0.880 | 0.213056 | 0.258854 | -0.045798 | 125 | 29 | 0.974026 | 0.993506 |
| recent utility last 5s | 11 | 0.063 | 0.001637 | 0.005582 | -0.003944 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 175 | 1.000 | 0.228374 | 0.270049 | -0.041675 | 144 | 31 | 0.977143 | 0.994286 |

## Active Smoke/Inferno Intervals

- `7.5s` - `84.0s`, rows `154`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.0`, LSTM `0.7047`, XGBoost `0.4078`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.3788`, XGBoost `0.1408`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.7622`, XGBoost `0.5416`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.6121`, XGBoost `0.4003`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.2932`, XGBoost `0.4620`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.2944`, XGBoost `0.4620`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.2977`, XGBoost `0.4634`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.5187`, XGBoost `0.3536`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.3025`, XGBoost `0.4674`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.3036`, XGBoost `0.4679`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
