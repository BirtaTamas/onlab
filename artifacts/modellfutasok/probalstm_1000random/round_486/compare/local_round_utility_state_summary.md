# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `18`
- rows: `213`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 213 | 1.000 | 0.485017 | 0.561250 | -0.076233 | 165 | 48 | 0.521127 | 0.366197 |
| active/recent utility | 213 | 1.000 | 0.485017 | 0.561250 | -0.076233 | 165 | 48 | 0.521127 | 0.366197 |
| strong utility action | 194 | 0.911 | 0.479605 | 0.555364 | -0.075759 | 146 | 48 | 0.510309 | 0.402062 |
| utility damage | 10 | 0.047 | 0.282895 | 0.481007 | -0.198112 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 184 | 0.864 | 0.481215 | 0.556944 | -0.075729 | 136 | 48 | 0.483696 | 0.423913 |
| recent utility last 5s | 30 | 0.141 | 0.536819 | 0.603067 | -0.066248 | 30 | 0 | 0.666667 | 0.333333 |
| flash effect present | 213 | 1.000 | 0.485017 | 0.561250 | -0.076233 | 165 | 48 | 0.521127 | 0.366197 |

## Active Smoke/Inferno Intervals

- `8.0s` - `77.5s`, rows `140`
- `83.5s` - `105.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.5`, LSTM `0.3010`, XGBoost `0.6964`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1067`, XGBoost `0.4840`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.1459`, XGBoost `0.4813`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.3930`, XGBoost `0.7276`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.1582`, XGBoost `0.4813`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1778`, XGBoost `0.4805`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.3651`, XGBoost `0.6675`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.1999`, XGBoost `0.4807`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.1066`, XGBoost `0.3862`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.2067`, XGBoost `0.4815`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `24.0`, recent_utility `0`
