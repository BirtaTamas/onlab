# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `15`
- rows: `203`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 203 | 1.000 | 0.518296 | 0.480263 | 0.038033 | 133 | 70 | 0.472906 | 0.295567 |
| active/recent utility | 203 | 1.000 | 0.518296 | 0.480263 | 0.038033 | 133 | 70 | 0.472906 | 0.295567 |
| strong utility action | 185 | 0.911 | 0.506760 | 0.461068 | 0.045692 | 126 | 59 | 0.421622 | 0.227027 |
| utility damage | 20 | 0.099 | 0.487048 | 0.505018 | -0.017970 | 4 | 16 | 0.200000 | 0.200000 |
| active smoke/inferno | 185 | 0.911 | 0.506760 | 0.461068 | 0.045692 | 126 | 59 | 0.421622 | 0.227027 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 203 | 1.000 | 0.518296 | 0.480263 | 0.038033 | 133 | 70 | 0.472906 | 0.295567 |

## Active Smoke/Inferno Intervals

- `8.0s` - `100.0s`, rows `185`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.5`, LSTM `0.5204`, XGBoost `0.3492`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5178`, XGBoost `0.3489`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.5153`, XGBoost `0.3492`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.5087`, XGBoost `0.3495`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5021`, XGBoost `0.3432`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5005`, XGBoost `0.3420`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.5092`, XGBoost `0.3512`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.4992`, XGBoost `0.3414`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.5061`, XGBoost `0.3486`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.4983`, XGBoost `0.3414`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
