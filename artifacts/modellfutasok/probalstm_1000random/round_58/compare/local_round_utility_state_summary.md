# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `5`
- rows: `303`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 303 | 1.000 | 0.392843 | 0.408421 | -0.015578 | 189 | 114 | 0.679868 | 0.884488 |
| active/recent utility | 303 | 1.000 | 0.392843 | 0.408421 | -0.015578 | 189 | 114 | 0.679868 | 0.884488 |
| strong utility action | 221 | 0.729 | 0.416754 | 0.433007 | -0.016252 | 128 | 93 | 0.669683 | 0.932127 |
| utility damage | 34 | 0.112 | 0.517265 | 0.571128 | -0.053863 | 26 | 8 | 0.529412 | 0.588235 |
| active smoke/inferno | 221 | 0.729 | 0.416754 | 0.433007 | -0.016252 | 128 | 93 | 0.669683 | 0.932127 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 303 | 1.000 | 0.392843 | 0.408421 | -0.015578 | 189 | 114 | 0.679868 | 0.884488 |

## Active Smoke/Inferno Intervals

- `10.0s` - `15.0s`, rows `11`
- `18.5s` - `101.0s`, rows `166`
- `114.5s` - `136.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.0`, LSTM `0.5785`, XGBoost `0.7409`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `80.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.5890`, XGBoost `0.7449`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `48.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.6026`, XGBoost `0.7449`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `64.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.6007`, XGBoost `0.7409`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `80.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.6109`, XGBoost `0.7400`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `121.0`, LSTM `0.1251`, XGBoost `0.2537`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.6136`, XGBoost `0.7400`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `120.5`, LSTM `0.1298`, XGBoost `0.2537`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `119.0`, LSTM `0.1400`, XGBoost `0.2613`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `119.5`, LSTM `0.1401`, XGBoost `0.2613`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
