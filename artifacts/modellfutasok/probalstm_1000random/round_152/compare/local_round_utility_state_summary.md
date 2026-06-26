# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `22`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.650976 | 0.617776 | 0.033200 | 66 | 91 | 0.872611 | 0.796178 |
| active/recent utility | 157 | 1.000 | 0.650976 | 0.617776 | 0.033200 | 66 | 91 | 0.872611 | 0.796178 |
| strong utility action | 95 | 0.605 | 0.548538 | 0.543434 | 0.005104 | 35 | 60 | 0.789474 | 0.852632 |
| utility damage | 22 | 0.140 | 0.467374 | 0.504022 | -0.036648 | 3 | 19 | 0.363636 | 0.818182 |
| active smoke/inferno | 85 | 0.541 | 0.550390 | 0.546017 | 0.004373 | 28 | 57 | 0.764706 | 0.835294 |
| recent utility last 5s | 30 | 0.191 | 0.551824 | 0.551146 | 0.000678 | 14 | 16 | 0.800000 | 1.000000 |
| flash effect present | 157 | 1.000 | 0.650976 | 0.617776 | 0.033200 | 66 | 91 | 0.872611 | 0.796178 |

## Active Smoke/Inferno Intervals

- `6.5s` - `48.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.0`, LSTM `0.5948`, XGBoost `0.3405`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5732`, XGBoost `0.3405`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5697`, XGBoost `0.3405`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5459`, XGBoost `0.3405`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5412`, XGBoost `0.3405`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5396`, XGBoost `0.3405`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5321`, XGBoost `0.3405`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5368`, XGBoost `0.3989`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6273`, XGBoost `0.7177`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6532`, XGBoost `0.7424`, closer `xgboost`, smoke `8`, inferno `1`, utility_damage `0.0`, recent_utility `0`
