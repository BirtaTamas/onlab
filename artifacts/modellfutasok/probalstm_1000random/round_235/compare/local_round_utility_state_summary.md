# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `17`
- rows: `218`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 218 | 1.000 | 0.253606 | 0.309703 | -0.056097 | 217 | 1 | 0.720183 | 0.628440 |
| active/recent utility | 218 | 1.000 | 0.253606 | 0.309703 | -0.056097 | 217 | 1 | 0.720183 | 0.628440 |
| strong utility action | 123 | 0.564 | 0.399366 | 0.490681 | -0.091315 | 122 | 1 | 0.504065 | 0.439024 |
| utility damage | 10 | 0.046 | 0.124183 | 0.243116 | -0.118933 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 123 | 0.564 | 0.399366 | 0.490681 | -0.091315 | 122 | 1 | 0.504065 | 0.439024 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 218 | 1.000 | 0.253606 | 0.309703 | -0.056097 | 217 | 1 | 0.720183 | 0.628440 |

## Active Smoke/Inferno Intervals

- `6.5s` - `67.5s`, rows `123`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.0192`, XGBoost `0.1822`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.0177`, XGBoost `0.1801`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.0238`, XGBoost `0.1817`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.0307`, XGBoost `0.1834`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.3224`, XGBoost `0.4723`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.0343`, XGBoost `0.1835`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.0388`, XGBoost `0.1841`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `37.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.1011`, XGBoost `0.2460`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1033`, XGBoost `0.2467`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `7.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.0445`, XGBoost `0.1871`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `37.0`, recent_utility `0`
