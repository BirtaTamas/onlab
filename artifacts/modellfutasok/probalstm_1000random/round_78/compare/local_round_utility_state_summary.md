# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `13`
- rows: `204`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.316860 | 0.317117 | -0.000257 | 97 | 107 | 0.882353 | 0.931373 |
| active/recent utility | 204 | 1.000 | 0.316860 | 0.317117 | -0.000257 | 97 | 107 | 0.882353 | 0.931373 |
| strong utility action | 88 | 0.431 | 0.286205 | 0.294911 | -0.008706 | 38 | 50 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.431 | 0.286205 | 0.294911 | -0.008706 | 38 | 50 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 204 | 1.000 | 0.316860 | 0.317117 | -0.000257 | 97 | 107 | 0.882353 | 0.931373 |

## Active Smoke/Inferno Intervals

- `35.0s` - `56.5s`, rows `44`
- `65.0s` - `86.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.0`, LSTM `0.0672`, XGBoost `0.2467`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.0734`, XGBoost `0.2487`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0752`, XGBoost `0.2493`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.4353`, XGBoost `0.2673`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.1065`, XGBoost `0.2675`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.0821`, XGBoost `0.2420`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.0911`, XGBoost `0.2493`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.1135`, XGBoost `0.2699`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.1136`, XGBoost `0.2693`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.1152`, XGBoost `0.2699`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
