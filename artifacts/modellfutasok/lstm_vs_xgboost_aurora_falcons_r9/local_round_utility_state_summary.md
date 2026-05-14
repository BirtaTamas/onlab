# Local Round Utility State Analysis

- csv_path: `processed_full\esports_world_cup\esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW\aurora-vs-falcons-m2-train.csv`
- round_num: `9`
- rows: `184`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.656428 | 0.751656 | -0.095229 | 5 | 179 | 0.679348 | 0.994565 |
| active/recent utility | 184 | 1.000 | 0.656428 | 0.751656 | -0.095229 | 5 | 179 | 0.679348 | 0.994565 |
| strong utility action | 148 | 0.804 | 0.597762 | 0.708407 | -0.110645 | 5 | 143 | 0.614865 | 1.000000 |
| utility damage | 10 | 0.054 | 0.656694 | 0.665642 | -0.008948 | 5 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 138 | 0.750 | 0.626585 | 0.722593 | -0.096008 | 5 | 133 | 0.659420 | 1.000000 |
| recent utility last 5s | 14 | 0.076 | 0.214164 | 0.517345 | -0.303181 | 0 | 14 | 0.000000 | 1.000000 |
| flash effect present | 184 | 1.000 | 0.656428 | 0.751656 | -0.095229 | 5 | 179 | 0.679348 | 0.994565 |

## Active Smoke/Inferno Intervals

- `6.0s` - `74.5s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `2.0`, LSTM `0.1825`, XGBoost `0.5139`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.1868`, XGBoost `0.5139`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.1892`, XGBoost `0.5147`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.2064`, XGBoost `0.5313`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `1.0`, LSTM `0.1994`, XGBoost `0.5145`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.1969`, XGBoost `0.5079`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.5`, LSTM `0.1996`, XGBoost `0.5076`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `5.0`, LSTM `0.2077`, XGBoost `0.5074`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.5`, LSTM `0.2332`, XGBoost `0.5308`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.2120`, XGBoost `0.5076`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
