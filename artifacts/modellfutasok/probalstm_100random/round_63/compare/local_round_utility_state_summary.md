# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `15`
- rows: `186`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 186 | 1.000 | 0.012515 | 0.038771 | -0.026257 | 186 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 186 | 1.000 | 0.012515 | 0.038771 | -0.026257 | 186 | 0 | 1.000000 | 1.000000 |
| strong utility action | 112 | 0.602 | 0.012865 | 0.034903 | -0.022038 | 112 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.054 | 0.017602 | 0.066165 | -0.048564 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 112 | 0.602 | 0.012865 | 0.034903 | -0.022038 | 112 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 186 | 1.000 | 0.012515 | 0.038771 | -0.026257 | 186 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `17.0s`, rows `14`
- `23.0s` - `30.0s`, rows `15`
- `36.0s` - `77.0s`, rows `83`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.0`, LSTM `0.0572`, XGBoost `0.1700`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0726`, XGBoost `0.1698`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0846`, XGBoost `0.1703`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1021`, XGBoost `0.1865`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0969`, XGBoost `0.1703`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1155`, XGBoost `0.1856`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0207`, XGBoost `0.0821`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1273`, XGBoost `0.1865`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0220`, XGBoost `0.0784`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `37.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.0106`, XGBoost `0.0633`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
