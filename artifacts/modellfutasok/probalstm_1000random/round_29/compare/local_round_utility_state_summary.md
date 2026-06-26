# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `9`
- rows: `270`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 270 | 1.000 | 0.166516 | 0.248569 | -0.082053 | 256 | 14 | 1.000000 | 0.981481 |
| active/recent utility | 270 | 1.000 | 0.166516 | 0.248569 | -0.082053 | 256 | 14 | 1.000000 | 0.981481 |
| strong utility action | 204 | 0.756 | 0.210695 | 0.306835 | -0.096140 | 190 | 14 | 1.000000 | 0.975490 |
| utility damage | 27 | 0.100 | 0.336459 | 0.430180 | -0.093721 | 19 | 8 | 1.000000 | 0.814815 |
| active smoke/inferno | 194 | 0.719 | 0.209076 | 0.302930 | -0.093854 | 180 | 14 | 1.000000 | 0.974227 |
| recent utility last 5s | 10 | 0.037 | 0.242105 | 0.382601 | -0.140496 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 270 | 1.000 | 0.166516 | 0.248569 | -0.082053 | 256 | 14 | 1.000000 | 0.981481 |

## Active Smoke/Inferno Intervals

- `6.5s` - `53.0s`, rows `94`
- `55.0s` - `61.5s`, rows `14`
- `63.0s` - `69.5s`, rows `14`
- `72.0s` - `107.5s`, rows `72`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `66.0`, LSTM `0.1681`, XGBoost `0.4170`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2545`, XGBoost `0.5013`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `46.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.1805`, XGBoost `0.4249`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.1735`, XGBoost `0.4170`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.1818`, XGBoost `0.4249`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.2566`, XGBoost `0.4928`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `46.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.1819`, XGBoost `0.4170`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.1915`, XGBoost `0.4170`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1920`, XGBoost `0.4170`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.2045`, XGBoost `0.4249`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
