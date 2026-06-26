# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `19`
- rows: `259`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 259 | 1.000 | 0.259546 | 0.291346 | -0.031800 | 192 | 67 | 0.694981 | 0.822394 |
| active/recent utility | 259 | 1.000 | 0.259546 | 0.291346 | -0.031800 | 192 | 67 | 0.694981 | 0.822394 |
| strong utility action | 186 | 0.718 | 0.356521 | 0.398661 | -0.042140 | 119 | 67 | 0.575269 | 0.752688 |
| utility damage | 20 | 0.077 | 0.264821 | 0.384424 | -0.119603 | 20 | 0 | 1.000000 | 0.500000 |
| active smoke/inferno | 177 | 0.683 | 0.355573 | 0.393486 | -0.037913 | 110 | 67 | 0.553672 | 0.757062 |
| recent utility last 5s | 10 | 0.039 | 0.372758 | 0.500105 | -0.127347 | 10 | 0 | 1.000000 | 0.700000 |
| flash effect present | 259 | 1.000 | 0.259546 | 0.291346 | -0.031800 | 192 | 67 | 0.694981 | 0.822394 |

## Active Smoke/Inferno Intervals

- `5.5s` - `93.5s`, rows `177`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `74.0`, LSTM `0.0493`, XGBoost `0.2495`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0500`, XGBoost `0.2475`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0712`, XGBoost `0.2534`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0724`, XGBoost `0.2534`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.0807`, XGBoost `0.2584`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.0759`, XGBoost `0.2534`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.0818`, XGBoost `0.2584`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0721`, XGBoost `0.2475`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.0795`, XGBoost `0.2544`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.0470`, XGBoost `0.2213`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `37.0`, recent_utility `0`
