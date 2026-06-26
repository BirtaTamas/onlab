# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `24`
- rows: `145`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 145 | 1.000 | 0.528382 | 0.561063 | -0.032681 | 56 | 89 | 0.717241 | 0.737931 |
| active/recent utility | 145 | 1.000 | 0.528382 | 0.561063 | -0.032681 | 56 | 89 | 0.717241 | 0.737931 |
| strong utility action | 140 | 0.966 | 0.529190 | 0.563177 | -0.033987 | 52 | 88 | 0.721429 | 0.757143 |
| utility damage | 34 | 0.234 | 0.627308 | 0.646996 | -0.019688 | 14 | 20 | 0.794118 | 0.823529 |
| active smoke/inferno | 127 | 0.876 | 0.522894 | 0.561062 | -0.038168 | 44 | 83 | 0.708661 | 0.740157 |
| recent utility last 5s | 21 | 0.145 | 0.653517 | 0.633255 | 0.020261 | 17 | 4 | 0.904762 | 0.952381 |
| flash effect present | 145 | 1.000 | 0.528382 | 0.561063 | -0.032681 | 56 | 89 | 0.717241 | 0.737931 |

## Active Smoke/Inferno Intervals

- `8.0s` - `71.0s`, rows `127`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.2323`, XGBoost `0.4441`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.2336`, XGBoost `0.4368`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5507`, XGBoost `0.7480`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.2810`, XGBoost `0.4544`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.2800`, XGBoost `0.4528`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.2819`, XGBoost `0.4536`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.2409`, XGBoost `0.4092`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.2377`, XGBoost `0.4039`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.3055`, XGBoost `0.4709`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5861`, XGBoost `0.7480`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
