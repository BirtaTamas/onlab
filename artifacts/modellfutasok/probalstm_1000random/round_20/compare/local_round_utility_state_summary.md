# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `13`
- rows: `183`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 183 | 1.000 | 0.740064 | 0.830982 | -0.090918 | 0 | 183 | 1.000000 | 1.000000 |
| active/recent utility | 183 | 1.000 | 0.740064 | 0.830982 | -0.090918 | 0 | 183 | 1.000000 | 1.000000 |
| strong utility action | 14 | 0.077 | 0.829042 | 0.922511 | -0.093470 | 0 | 14 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 14 | 0.077 | 0.829042 | 0.922511 | -0.093470 | 0 | 14 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 183 | 1.000 | 0.740064 | 0.830982 | -0.090918 | 0 | 183 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `66.0s` - `72.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.5`, LSTM `0.7845`, XGBoost `0.9016`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7170`, XGBoost `0.8323`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.8154`, XGBoost `0.9247`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.8162`, XGBoost `0.9242`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.8182`, XGBoost `0.9247`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.7308`, XGBoost `0.8333`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.8259`, XGBoost `0.9239`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.8291`, XGBoost `0.9251`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.8205`, XGBoost `0.9159`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.8305`, XGBoost `0.9253`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
