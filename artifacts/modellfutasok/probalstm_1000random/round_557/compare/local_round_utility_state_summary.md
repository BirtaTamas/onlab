# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-falcons-vs-vitality-bo3-948Z-JwufPJ8ROXkhPE5QF/falcons-vs-vitality-m2-nuke.csv`
- round_num: `4`
- rows: `250`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 250 | 1.000 | 0.337009 | 0.437429 | -0.100420 | 247 | 3 | 0.624000 | 0.452000 |
| active/recent utility | 250 | 1.000 | 0.337009 | 0.437429 | -0.100420 | 247 | 3 | 0.624000 | 0.452000 |
| strong utility action | 179 | 0.716 | 0.394050 | 0.511738 | -0.117688 | 177 | 2 | 0.508380 | 0.357542 |
| utility damage | 10 | 0.040 | 0.746266 | 0.822213 | -0.075947 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 179 | 0.716 | 0.394050 | 0.511738 | -0.117688 | 177 | 2 | 0.508380 | 0.357542 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 250 | 1.000 | 0.337009 | 0.437429 | -0.100420 | 247 | 3 | 0.624000 | 0.452000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `50.0s`, rows `85`
- `53.5s` - `100.0s`, rows `94`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.6427`, XGBoost `0.8552`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6430`, XGBoost `0.8555`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.5381`, XGBoost `0.7486`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6475`, XGBoost `0.8544`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.6501`, XGBoost `0.8559`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.6530`, XGBoost `0.8543`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.6541`, XGBoost `0.8543`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.6546`, XGBoost `0.8544`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.6581`, XGBoost `0.8543`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.5605`, XGBoost `0.7563`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
