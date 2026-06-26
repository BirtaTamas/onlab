# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `20`
- rows: `196`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.228655 | 0.273657 | -0.045002 | 23 | 173 | 0.107143 | 0.219388 |
| active/recent utility | 196 | 1.000 | 0.228655 | 0.273657 | -0.045002 | 23 | 173 | 0.107143 | 0.219388 |
| strong utility action | 146 | 0.745 | 0.235296 | 0.282770 | -0.047474 | 23 | 123 | 0.130137 | 0.171233 |
| utility damage | 61 | 0.311 | 0.372217 | 0.397158 | -0.024941 | 15 | 46 | 0.311475 | 0.377049 |
| active smoke/inferno | 137 | 0.699 | 0.192629 | 0.242551 | -0.049922 | 20 | 117 | 0.072993 | 0.116788 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 196 | 1.000 | 0.228655 | 0.273657 | -0.045002 | 23 | 173 | 0.107143 | 0.219388 |

## Active Smoke/Inferno Intervals

- `8.0s` - `70.5s`, rows `126`
- `87.0s` - `92.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.2525`, XGBoost `0.4419`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.3292`, XGBoost `0.1453`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.2722`, XGBoost `0.4485`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.0546`, XGBoost `0.2287`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.3043`, XGBoost `0.4782`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `63.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.0538`, XGBoost `0.2266`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.0552`, XGBoost `0.2264`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.0568`, XGBoost `0.2266`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0552`, XGBoost `0.2160`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0605`, XGBoost `0.2157`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
