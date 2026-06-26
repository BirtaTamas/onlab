# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `10`
- rows: `191`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.466917 | 0.434928 | 0.031990 | 155 | 36 | 0.465969 | 0.345550 |
| active/recent utility | 191 | 1.000 | 0.466917 | 0.434928 | 0.031990 | 155 | 36 | 0.465969 | 0.345550 |
| strong utility action | 180 | 0.942 | 0.462521 | 0.431300 | 0.031221 | 145 | 35 | 0.438889 | 0.311111 |
| utility damage | 11 | 0.058 | 0.269443 | 0.278635 | -0.009192 | 5 | 6 | 0.000000 | 0.000000 |
| active smoke/inferno | 170 | 0.890 | 0.457462 | 0.426212 | 0.031250 | 135 | 35 | 0.405882 | 0.270588 |
| recent utility last 5s | 10 | 0.052 | 0.548539 | 0.517808 | 0.030731 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 191 | 1.000 | 0.466917 | 0.434928 | 0.031990 | 155 | 36 | 0.465969 | 0.345550 |

## Active Smoke/Inferno Intervals

- `10.0s` - `24.5s`, rows `30`
- `25.5s` - `95.0s`, rows `140`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.0`, LSTM `0.4092`, XGBoost `0.2950`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4029`, XGBoost `0.2936`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6250`, XGBoost `0.5194`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.3999`, XGBoost `0.2962`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.6326`, XGBoost `0.5312`, closer `lstm`, smoke `0`, inferno `4`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.3956`, XGBoost `0.2950`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6333`, XGBoost `0.5327`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.2961`, XGBoost `0.1987`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `48.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.3890`, XGBoost `0.2950`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1802`, XGBoost `0.0872`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
