# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `4`
- rows: `261`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 261 | 1.000 | 0.052970 | 0.157611 | -0.104641 | 261 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 261 | 1.000 | 0.052970 | 0.157611 | -0.104641 | 261 | 0 | 1.000000 | 1.000000 |
| strong utility action | 184 | 0.705 | 0.061995 | 0.188815 | -0.126820 | 184 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 174 | 0.667 | 0.061070 | 0.185178 | -0.124107 | 174 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.038 | 0.078087 | 0.252104 | -0.174018 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 261 | 1.000 | 0.052970 | 0.157611 | -0.104641 | 261 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `13.0s` - `34.5s`, rows `44`
- `40.0s` - `104.5s`, rows `130`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.0433`, XGBoost `0.2938`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.0413`, XGBoost `0.2869`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.0448`, XGBoost `0.2869`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.0500`, XGBoost `0.2901`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.0500`, XGBoost `0.2895`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.0438`, XGBoost `0.2823`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0524`, XGBoost `0.2901`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.0454`, XGBoost `0.2823`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0533`, XGBoost `0.2901`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0565`, XGBoost `0.2865`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
