# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `4`
- rows: `255`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 255 | 1.000 | 0.393619 | 0.545510 | -0.151891 | 250 | 5 | 0.631373 | 0.192157 |
| active/recent utility | 255 | 1.000 | 0.393619 | 0.545510 | -0.151891 | 250 | 5 | 0.631373 | 0.192157 |
| strong utility action | 192 | 0.753 | 0.397859 | 0.524078 | -0.126219 | 187 | 5 | 0.583333 | 0.203125 |
| utility damage | 20 | 0.078 | 0.417936 | 0.562646 | -0.144710 | 15 | 5 | 0.500000 | 0.050000 |
| active smoke/inferno | 192 | 0.753 | 0.397859 | 0.524078 | -0.126219 | 187 | 5 | 0.583333 | 0.203125 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 255 | 1.000 | 0.393619 | 0.545510 | -0.151891 | 250 | 5 | 0.631373 | 0.192157 |

## Active Smoke/Inferno Intervals

- `7.0s` - `102.5s`, rows `192`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.0`, LSTM `0.1985`, XGBoost `0.5719`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.2137`, XGBoost `0.5769`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.2218`, XGBoost `0.5803`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.2180`, XGBoost `0.5756`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.2169`, XGBoost `0.5719`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.2199`, XGBoost `0.5719`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.2222`, XGBoost `0.5738`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.2336`, XGBoost `0.5803`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.2276`, XGBoost `0.5719`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.2285`, XGBoost `0.5719`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
