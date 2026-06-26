# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `9`
- rows: `157`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.030785 | 0.066684 | -0.035899 | 136 | 21 | 1.000000 | 1.000000 |
| active/recent utility | 157 | 1.000 | 0.030785 | 0.066684 | -0.035899 | 136 | 21 | 1.000000 | 1.000000 |
| strong utility action | 109 | 0.694 | 0.015560 | 0.027119 | -0.011559 | 106 | 3 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 109 | 0.694 | 0.015560 | 0.027119 | -0.011559 | 106 | 3 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 157 | 1.000 | 0.030785 | 0.066684 | -0.035899 | 136 | 21 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `13.5s` - `67.5s`, rows `109`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.0`, LSTM `0.0799`, XGBoost `0.1664`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0853`, XGBoost `0.1664`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0846`, XGBoost `0.1632`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0853`, XGBoost `0.1632`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0870`, XGBoost `0.1632`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0930`, XGBoost `0.1664`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0919`, XGBoost `0.1632`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.0888`, XGBoost `0.1575`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0933`, XGBoost `0.1575`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0973`, XGBoost `0.1607`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
