# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `19`
- rows: `182`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.884242 | 0.865313 | 0.018928 | 73 | 109 | 1.000000 | 1.000000 |
| active/recent utility | 182 | 1.000 | 0.884242 | 0.865313 | 0.018928 | 73 | 109 | 1.000000 | 1.000000 |
| strong utility action | 160 | 0.879 | 0.894675 | 0.875480 | 0.019195 | 60 | 100 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.055 | 0.763220 | 0.676141 | 0.087080 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 160 | 0.879 | 0.894675 | 0.875480 | 0.019195 | 60 | 100 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.055 | 0.972255 | 0.979779 | -0.007524 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 182 | 1.000 | 0.884242 | 0.865313 | 0.018928 | 73 | 109 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `86.0s`, rows `160`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.0`, LSTM `0.8030`, XGBoost `0.6861`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.7985`, XGBoost `0.6850`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.7984`, XGBoost `0.6850`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7942`, XGBoost `0.6861`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.7831`, XGBoost `0.6754`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.7825`, XGBoost `0.6754`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.7816`, XGBoost `0.6754`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.7788`, XGBoost `0.6754`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7886`, XGBoost `0.6861`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.7876`, XGBoost `0.6871`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
