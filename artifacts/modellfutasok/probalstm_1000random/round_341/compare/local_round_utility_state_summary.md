# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `5`
- rows: `168`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 168 | 1.000 | 0.601749 | 0.655315 | -0.053566 | 0 | 168 | 0.636905 | 0.779762 |
| active/recent utility | 168 | 1.000 | 0.601749 | 0.655315 | -0.053566 | 0 | 168 | 0.636905 | 0.779762 |
| strong utility action | 139 | 0.827 | 0.567409 | 0.628097 | -0.060688 | 0 | 139 | 0.647482 | 0.820144 |
| utility damage | 20 | 0.119 | 0.474990 | 0.531425 | -0.056435 | 0 | 20 | 0.250000 | 0.500000 |
| active smoke/inferno | 139 | 0.827 | 0.567409 | 0.628097 | -0.060688 | 0 | 139 | 0.647482 | 0.820144 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 168 | 1.000 | 0.601749 | 0.655315 | -0.053566 | 0 | 168 | 0.636905 | 0.779762 |

## Active Smoke/Inferno Intervals

- `6.0s` - `75.0s`, rows `139`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.6713`, XGBoost `0.8012`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.4522`, XGBoost `0.5702`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.4518`, XGBoost `0.5694`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `18.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.6922`, XGBoost `0.8024`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.6938`, XGBoost `0.8024`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.6926`, XGBoost `0.8007`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.6959`, XGBoost `0.8024`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.4697`, XGBoost `0.5743`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.4671`, XGBoost `0.5683`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `18.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.7032`, XGBoost `0.8024`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
