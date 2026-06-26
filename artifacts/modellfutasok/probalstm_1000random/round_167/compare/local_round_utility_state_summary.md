# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-g2-vs-betboom-bo3-pCfbtiY01aL_JW2Hy1pnZ6/g2-vs-betboom-m1-anubis.csv`
- round_num: `8`
- rows: `221`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 221 | 1.000 | 0.341114 | 0.381680 | -0.040566 | 127 | 94 | 0.542986 | 0.592760 |
| active/recent utility | 221 | 1.000 | 0.341114 | 0.381680 | -0.040566 | 127 | 94 | 0.542986 | 0.592760 |
| strong utility action | 143 | 0.647 | 0.324969 | 0.383722 | -0.058753 | 92 | 51 | 0.601399 | 0.622378 |
| utility damage | 10 | 0.045 | 0.177015 | 0.330106 | -0.153091 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 143 | 0.647 | 0.324969 | 0.383722 | -0.058753 | 92 | 51 | 0.601399 | 0.622378 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 221 | 1.000 | 0.341114 | 0.381680 | -0.040566 | 127 | 94 | 0.542986 | 0.592760 |

## Active Smoke/Inferno Intervals

- `8.0s` - `72.0s`, rows `129`
- `94.5s` - `101.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.1244`, XGBoost `0.3290`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1343`, XGBoost `0.3273`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1447`, XGBoost `0.3346`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1480`, XGBoost `0.3346`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1485`, XGBoost `0.3346`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1425`, XGBoost `0.3273`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1501`, XGBoost `0.3304`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.1558`, XGBoost `0.3346`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1481`, XGBoost `0.3249`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.1578`, XGBoost `0.3346`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
