# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `4`
- rows: `189`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.773888 | 0.760811 | 0.013077 | 93 | 96 | 1.000000 | 1.000000 |
| active/recent utility | 189 | 1.000 | 0.773888 | 0.760811 | 0.013077 | 93 | 96 | 1.000000 | 1.000000 |
| strong utility action | 166 | 0.878 | 0.779779 | 0.771832 | 0.007947 | 72 | 94 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.053 | 0.676542 | 0.614767 | 0.061775 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 166 | 0.878 | 0.779779 | 0.771832 | 0.007947 | 72 | 94 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.053 | 0.728709 | 0.802343 | -0.073634 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 189 | 1.000 | 0.773888 | 0.760811 | 0.013077 | 93 | 96 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `50.5s`, rows `81`
- `52.0s` - `94.0s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.5`, LSTM `0.8253`, XGBoost `0.6334`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.7391`, XGBoost `0.5514`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.8202`, XGBoost `0.6334`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.8074`, XGBoost `0.6226`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.7977`, XGBoost `0.6164`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.7954`, XGBoost `0.6164`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.7891`, XGBoost `0.6142`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.8082`, XGBoost `0.6334`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.7942`, XGBoost `0.6226`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.7927`, XGBoost `0.6226`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
