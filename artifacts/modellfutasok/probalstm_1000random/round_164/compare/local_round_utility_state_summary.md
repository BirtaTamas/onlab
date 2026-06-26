# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m3-mirage.csv`
- round_num: `10`
- rows: `293`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 293 | 1.000 | 0.623441 | 0.644252 | -0.020811 | 133 | 160 | 0.194539 | 0.160410 |
| active/recent utility | 293 | 1.000 | 0.623441 | 0.644252 | -0.020811 | 133 | 160 | 0.194539 | 0.160410 |
| strong utility action | 206 | 0.703 | 0.672448 | 0.675631 | -0.003183 | 72 | 134 | 0.106796 | 0.092233 |
| utility damage | 21 | 0.072 | 0.679267 | 0.623446 | 0.055821 | 4 | 17 | 0.000000 | 0.000000 |
| active smoke/inferno | 200 | 0.683 | 0.671845 | 0.677713 | -0.005868 | 72 | 128 | 0.110000 | 0.095000 |
| recent utility last 5s | 20 | 0.068 | 0.706942 | 0.665275 | 0.041666 | 9 | 11 | 0.000000 | 0.000000 |
| flash effect present | 293 | 1.000 | 0.623441 | 0.644252 | -0.020811 | 133 | 160 | 0.194539 | 0.160410 |

## Active Smoke/Inferno Intervals

- `5.5s` - `61.0s`, rows `112`
- `65.5s` - `87.0s`, rows `44`
- `100.5s` - `122.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `110.0`, LSTM `0.4288`, XGBoost `0.8385`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.4676`, XGBoost `0.8145`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.1276`, XGBoost `0.4713`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.4694`, XGBoost `0.8129`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.1744`, XGBoost `0.4713`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.2160`, XGBoost `0.4713`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `113.0`, LSTM `0.6072`, XGBoost `0.8268`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.2810`, XGBoost `0.4713`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.6301`, XGBoost `0.8145`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.6457`, XGBoost `0.8268`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
