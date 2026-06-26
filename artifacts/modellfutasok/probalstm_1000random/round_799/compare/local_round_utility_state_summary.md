# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `3`
- rows: `133`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.174944 | 0.218666 | -0.043722 | 133 | 0 | 1.000000 | 0.819549 |
| active/recent utility | 133 | 1.000 | 0.174944 | 0.218666 | -0.043722 | 133 | 0 | 1.000000 | 0.819549 |
| strong utility action | 93 | 0.699 | 0.181259 | 0.238634 | -0.057375 | 93 | 0 | 1.000000 | 0.827957 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 93 | 0.699 | 0.181259 | 0.238634 | -0.057375 | 93 | 0 | 1.000000 | 0.827957 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 133 | 1.000 | 0.174944 | 0.218666 | -0.043722 | 133 | 0 | 1.000000 | 0.819549 |

## Active Smoke/Inferno Intervals

- `6.5s` - `52.5s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.2250`, XGBoost `0.4252`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `18.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2443`, XGBoost `0.4420`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2487`, XGBoost `0.4310`, closer `lstm`, smoke `6`, inferno `2`, utility_damage `18.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.2505`, XGBoost `0.4283`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.2744`, XGBoost `0.4315`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.2770`, XGBoost `0.4299`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.3564`, XGBoost `0.5005`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.3128`, XGBoost `0.4256`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.0295`, XGBoost `0.1423`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.3753`, XGBoost `0.4881`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
