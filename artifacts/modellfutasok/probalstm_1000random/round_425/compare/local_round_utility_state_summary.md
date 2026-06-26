# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `31`
- rows: `179`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 179 | 1.000 | 0.755931 | 0.813728 | -0.057798 | 13 | 166 | 0.977654 | 0.921788 |
| active/recent utility | 179 | 1.000 | 0.755931 | 0.813728 | -0.057798 | 13 | 166 | 0.977654 | 0.921788 |
| strong utility action | 168 | 0.939 | 0.754491 | 0.814769 | -0.060278 | 9 | 159 | 0.976190 | 0.928571 |
| utility damage | 27 | 0.151 | 0.671085 | 0.737792 | -0.066706 | 0 | 27 | 1.000000 | 1.000000 |
| active smoke/inferno | 158 | 0.883 | 0.770368 | 0.834692 | -0.064324 | 1 | 157 | 0.987342 | 0.981013 |
| recent utility last 5s | 10 | 0.056 | 0.503632 | 0.499976 | 0.003657 | 8 | 2 | 0.800000 | 0.100000 |
| flash effect present | 179 | 1.000 | 0.755931 | 0.813728 | -0.057798 | 13 | 166 | 0.977654 | 0.921788 |

## Active Smoke/Inferno Intervals

- `7.0s` - `63.5s`, rows `114`
- `64.5s` - `86.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.5549`, XGBoost `0.7343`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5701`, XGBoost `0.7341`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5717`, XGBoost `0.7341`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5739`, XGBoost `0.7341`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5760`, XGBoost `0.7358`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5832`, XGBoost `0.7341`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5917`, XGBoost `0.7350`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5921`, XGBoost `0.7341`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5922`, XGBoost `0.7341`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5972`, XGBoost `0.7338`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
