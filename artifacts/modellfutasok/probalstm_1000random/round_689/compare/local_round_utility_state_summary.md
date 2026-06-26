# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `1`
- rows: `181`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 181 | 1.000 | 0.239610 | 0.248761 | -0.009151 | 101 | 80 | 0.635359 | 0.994475 |
| active/recent utility | 181 | 1.000 | 0.239610 | 0.248761 | -0.009151 | 101 | 80 | 0.635359 | 0.994475 |
| strong utility action | 109 | 0.602 | 0.162369 | 0.179010 | -0.016642 | 75 | 34 | 0.798165 | 0.990826 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 109 | 0.602 | 0.162369 | 0.179010 | -0.016642 | 75 | 34 | 0.798165 | 0.990826 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 181 | 1.000 | 0.239610 | 0.248761 | -0.009151 | 101 | 80 | 0.635359 | 0.994475 |

## Active Smoke/Inferno Intervals

- `23.0s` - `77.0s`, rows `109`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `68.0`, LSTM `0.3502`, XGBoost `0.5554`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.3212`, XGBoost `0.1673`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1161`, XGBoost `0.2641`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.1239`, XGBoost `0.2641`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.1265`, XGBoost `0.2576`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5434`, XGBoost `0.4241`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5360`, XGBoost `0.4218`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0513`, XGBoost `0.1649`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0518`, XGBoost `0.1649`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0552`, XGBoost `0.1649`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
