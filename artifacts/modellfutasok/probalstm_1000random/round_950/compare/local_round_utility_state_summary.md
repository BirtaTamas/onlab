# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `10`
- rows: `188`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 188 | 1.000 | 0.101189 | 0.124813 | -0.023624 | 188 | 0 | 0.978723 | 0.829787 |
| active/recent utility | 188 | 1.000 | 0.101189 | 0.124813 | -0.023624 | 188 | 0 | 0.978723 | 0.829787 |
| strong utility action | 156 | 0.830 | 0.060949 | 0.081996 | -0.021046 | 156 | 0 | 0.980769 | 0.923077 |
| utility damage | 10 | 0.053 | 0.437235 | 0.466026 | -0.028791 | 10 | 0 | 0.900000 | 0.200000 |
| active smoke/inferno | 156 | 0.830 | 0.060949 | 0.081996 | -0.021046 | 156 | 0 | 0.980769 | 0.923077 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 188 | 1.000 | 0.101189 | 0.124813 | -0.023624 | 188 | 0 | 0.978723 | 0.829787 |

## Active Smoke/Inferno Intervals

- `10.0s` - `68.5s`, rows `118`
- `75.0s` - `93.5s`, rows `38`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.0`, LSTM `0.0451`, XGBoost `0.2102`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0391`, XGBoost `0.1944`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.0847`, XGBoost `0.1589`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1865`, XGBoost `0.2562`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.2118`, XGBoost `0.2814`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0259`, XGBoost `0.0915`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0534`, XGBoost `0.1188`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1928`, XGBoost `0.2575`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1948`, XGBoost `0.2552`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0274`, XGBoost `0.0847`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `26.0`, recent_utility `0`
