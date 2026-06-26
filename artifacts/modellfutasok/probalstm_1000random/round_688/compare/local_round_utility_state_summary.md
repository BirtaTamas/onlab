# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-Z9VnvF_JkEDX6y_HyMsFXx/aurora-vs-heroic-m3-mirage.csv`
- round_num: `18`
- rows: `200`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.162466 | 0.341170 | -0.178704 | 200 | 0 | 1.000000 | 0.985000 |
| active/recent utility | 200 | 1.000 | 0.162466 | 0.341170 | -0.178704 | 200 | 0 | 1.000000 | 0.985000 |
| strong utility action | 131 | 0.655 | 0.192993 | 0.374262 | -0.181269 | 131 | 0 | 1.000000 | 0.977099 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 131 | 0.655 | 0.192993 | 0.374262 | -0.181269 | 131 | 0 | 1.000000 | 0.977099 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 200 | 1.000 | 0.162466 | 0.341170 | -0.178704 | 200 | 0 | 1.000000 | 0.985000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `41.5s`, rows `68`
- `58.0s` - `89.0s`, rows `63`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.1043`, XGBoost `0.4317`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1089`, XGBoost `0.4309`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1128`, XGBoost `0.4079`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.1437`, XGBoost `0.4305`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.1300`, XGBoost `0.4079`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.1286`, XGBoost `0.4042`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1306`, XGBoost `0.4032`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.1357`, XGBoost `0.4037`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.1427`, XGBoost `0.4079`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.1409`, XGBoost `0.4057`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
