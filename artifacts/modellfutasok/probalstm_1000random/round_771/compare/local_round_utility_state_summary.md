# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m1-mirage.csv`
- round_num: `4`
- rows: `117`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 117 | 1.000 | 0.345195 | 0.471936 | -0.126741 | 102 | 15 | 0.529915 | 0.435897 |
| active/recent utility | 117 | 1.000 | 0.345195 | 0.471936 | -0.126741 | 102 | 15 | 0.529915 | 0.435897 |
| strong utility action | 85 | 0.726 | 0.444952 | 0.582603 | -0.137651 | 72 | 13 | 0.400000 | 0.270588 |
| utility damage | 28 | 0.239 | 0.535888 | 0.551133 | -0.015245 | 18 | 10 | 0.178571 | 0.178571 |
| active smoke/inferno | 75 | 0.641 | 0.431148 | 0.585838 | -0.154689 | 65 | 10 | 0.453333 | 0.306667 |
| recent utility last 5s | 10 | 0.085 | 0.548479 | 0.558343 | -0.009865 | 7 | 3 | 0.000000 | 0.000000 |
| flash effect present | 117 | 1.000 | 0.345195 | 0.471936 | -0.126741 | 102 | 15 | 0.529915 | 0.435897 |

## Active Smoke/Inferno Intervals

- `7.0s` - `44.0s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.1979`, XGBoost `0.6539`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.2047`, XGBoost `0.6421`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.2359`, XGBoost `0.6552`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.2403`, XGBoost `0.6552`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.2165`, XGBoost `0.6244`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.2508`, XGBoost `0.6529`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.0697`, XGBoost `0.4663`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.0739`, XGBoost `0.4663`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1266`, XGBoost `0.4916`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1253`, XGBoost `0.4902`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
