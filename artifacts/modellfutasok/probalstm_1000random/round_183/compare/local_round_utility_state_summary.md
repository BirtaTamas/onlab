# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `20`
- rows: `227`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 227 | 1.000 | 0.876814 | 0.908828 | -0.032014 | 2 | 225 | 1.000000 | 1.000000 |
| active/recent utility | 227 | 1.000 | 0.876814 | 0.908828 | -0.032014 | 2 | 225 | 1.000000 | 1.000000 |
| strong utility action | 185 | 0.815 | 0.862490 | 0.896410 | -0.033920 | 0 | 185 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 185 | 0.815 | 0.862490 | 0.896410 | -0.033920 | 0 | 185 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.088 | 0.867167 | 0.887008 | -0.019841 | 0 | 20 | 1.000000 | 1.000000 |
| flash effect present | 227 | 1.000 | 0.876814 | 0.908828 | -0.032014 | 2 | 225 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `47.0s`, rows `89`
- `52.5s` - `75.5s`, rows `47`
- `76.5s` - `99.5s`, rows `47`
- `112.5s` - `113.0s`, rows `2`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.5`, LSTM `0.8702`, XGBoost `0.9645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.8742`, XGBoost `0.9645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.8748`, XGBoost `0.9640`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.8767`, XGBoost `0.9645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.8763`, XGBoost `0.9641`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.8835`, XGBoost `0.9645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.8865`, XGBoost `0.9664`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.8843`, XGBoost `0.9640`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.8892`, XGBoost `0.9645`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.8862`, XGBoost `0.9609`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
