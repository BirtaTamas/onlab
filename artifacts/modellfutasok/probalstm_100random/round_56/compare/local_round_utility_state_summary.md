# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `14`
- rows: `159`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 159 | 1.000 | 0.459460 | 0.503989 | -0.044529 | 144 | 15 | 0.672956 | 0.672956 |
| active/recent utility | 159 | 1.000 | 0.459460 | 0.503989 | -0.044529 | 144 | 15 | 0.672956 | 0.672956 |
| strong utility action | 128 | 0.805 | 0.505912 | 0.550408 | -0.044496 | 117 | 11 | 0.593750 | 0.593750 |
| utility damage | 10 | 0.063 | 0.277357 | 0.345909 | -0.068552 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 120 | 0.755 | 0.521403 | 0.564062 | -0.042659 | 109 | 11 | 0.566667 | 0.566667 |
| recent utility last 5s | 10 | 0.063 | 0.965766 | 0.967148 | -0.001382 | 4 | 6 | 0.000000 | 0.000000 |
| flash effect present | 159 | 1.000 | 0.459460 | 0.503989 | -0.044529 | 144 | 15 | 0.672956 | 0.672956 |

## Active Smoke/Inferno Intervals

- `10.5s` - `34.5s`, rows `49`
- `42.5s` - `77.5s`, rows `71`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.2223`, XGBoost `0.3511`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.2268`, XGBoost `0.3293`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.2225`, XGBoost `0.3239`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.2194`, XGBoost `0.3162`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.2552`, XGBoost `0.3518`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.2193`, XGBoost `0.3157`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.2193`, XGBoost `0.3155`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.2294`, XGBoost `0.3244`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.2320`, XGBoost `0.3267`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.2313`, XGBoost `0.3239`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
