# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `17`
- rows: `141`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 141 | 1.000 | 0.168621 | 0.194239 | -0.025618 | 120 | 21 | 0.730496 | 0.730496 |
| active/recent utility | 141 | 1.000 | 0.168621 | 0.194239 | -0.025618 | 120 | 21 | 0.730496 | 0.730496 |
| strong utility action | 126 | 0.894 | 0.122072 | 0.150537 | -0.028465 | 112 | 14 | 0.817460 | 0.817460 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 126 | 0.894 | 0.122072 | 0.150537 | -0.028465 | 112 | 14 | 0.817460 | 0.817460 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 141 | 1.000 | 0.168621 | 0.194239 | -0.025618 | 120 | 21 | 0.730496 | 0.730496 |

## Active Smoke/Inferno Intervals

- `7.5s` - `70.0s`, rows `126`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.5`, LSTM `0.0398`, XGBoost `0.1380`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.0220`, XGBoost `0.1163`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0226`, XGBoost `0.1163`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0321`, XGBoost `0.1247`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.0264`, XGBoost `0.1184`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0309`, XGBoost `0.1215`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0294`, XGBoost `0.1199`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0284`, XGBoost `0.1184`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0314`, XGBoost `0.1209`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0238`, XGBoost `0.1132`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
