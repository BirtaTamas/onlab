# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `14`
- rows: `123`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.940233 | 0.985244 | -0.045011 | 0 | 123 | 1.000000 | 1.000000 |
| active/recent utility | 123 | 1.000 | 0.940233 | 0.985244 | -0.045011 | 0 | 123 | 1.000000 | 1.000000 |
| strong utility action | 69 | 0.561 | 0.936091 | 0.984758 | -0.048667 | 0 | 69 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 69 | 0.561 | 0.936091 | 0.984758 | -0.048667 | 0 | 69 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 123 | 1.000 | 0.940233 | 0.985244 | -0.045011 | 0 | 123 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `43.0s`, rows `69`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.0`, LSTM `0.9036`, XGBoost `0.9801`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.9041`, XGBoost `0.9791`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.9066`, XGBoost `0.9801`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.9086`, XGBoost `0.9791`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9093`, XGBoost `0.9791`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9104`, XGBoost `0.9790`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.9109`, XGBoost `0.9794`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.9116`, XGBoost `0.9791`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.9121`, XGBoost `0.9791`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.9143`, XGBoost `0.9794`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
