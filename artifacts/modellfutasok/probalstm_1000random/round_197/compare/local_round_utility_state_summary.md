# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `8`
- rows: `175`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 175 | 1.000 | 0.101000 | 0.188995 | -0.087995 | 175 | 0 | 1.000000 | 0.828571 |
| active/recent utility | 175 | 1.000 | 0.101000 | 0.188995 | -0.087995 | 175 | 0 | 1.000000 | 0.828571 |
| strong utility action | 99 | 0.566 | 0.104053 | 0.216790 | -0.112736 | 99 | 0 | 1.000000 | 0.848485 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 99 | 0.566 | 0.104053 | 0.216790 | -0.112736 | 99 | 0 | 1.000000 | 0.848485 |
| recent utility last 5s | 10 | 0.057 | 0.149109 | 0.258667 | -0.109558 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 175 | 1.000 | 0.101000 | 0.188995 | -0.087995 | 175 | 0 | 1.000000 | 0.828571 |

## Active Smoke/Inferno Intervals

- `7.5s` - `54.0s`, rows `94`
- `85.0s` - `87.0s`, rows `5`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.5`, LSTM `0.0704`, XGBoost `0.3004`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.0762`, XGBoost `0.3028`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.0774`, XGBoost `0.3033`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.0766`, XGBoost `0.3025`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0787`, XGBoost `0.3022`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0765`, XGBoost `0.2996`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0791`, XGBoost `0.2996`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.0805`, XGBoost `0.2996`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0786`, XGBoost `0.2947`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0939`, XGBoost `0.3086`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
