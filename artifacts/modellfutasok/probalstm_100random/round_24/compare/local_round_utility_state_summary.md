# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `9`
- rows: `188`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 188 | 1.000 | 0.158198 | 0.220980 | -0.062782 | 182 | 6 | 0.877660 | 0.723404 |
| active/recent utility | 188 | 1.000 | 0.158198 | 0.220980 | -0.062782 | 182 | 6 | 0.877660 | 0.723404 |
| strong utility action | 115 | 0.612 | 0.234376 | 0.310085 | -0.075709 | 109 | 6 | 0.800000 | 0.582609 |
| utility damage | 21 | 0.112 | 0.488038 | 0.519522 | -0.031484 | 17 | 4 | 0.380952 | 0.238095 |
| active smoke/inferno | 105 | 0.559 | 0.212564 | 0.289618 | -0.077054 | 99 | 6 | 0.780952 | 0.638095 |
| recent utility last 5s | 10 | 0.053 | 0.463404 | 0.524994 | -0.061590 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 188 | 1.000 | 0.158198 | 0.220980 | -0.062782 | 182 | 6 | 0.877660 | 0.723404 |

## Active Smoke/Inferno Intervals

- `7.0s` - `31.0s`, rows `49`
- `38.5s` - `66.0s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.5`, LSTM `0.0162`, XGBoost `0.2197`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0175`, XGBoost `0.2209`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0156`, XGBoost `0.2163`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0169`, XGBoost `0.2156`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.0167`, XGBoost `0.2141`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0228`, XGBoost `0.2196`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0275`, XGBoost `0.2219`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0295`, XGBoost `0.2220`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0353`, XGBoost `0.2232`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0362`, XGBoost `0.2230`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `10.0`, recent_utility `0`
