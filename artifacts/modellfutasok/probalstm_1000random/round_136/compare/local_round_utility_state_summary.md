# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `31`
- rows: `249`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 249 | 1.000 | 0.629771 | 0.603497 | 0.026275 | 161 | 88 | 0.843373 | 0.566265 |
| active/recent utility | 249 | 1.000 | 0.629771 | 0.603497 | 0.026275 | 161 | 88 | 0.843373 | 0.566265 |
| strong utility action | 148 | 0.594 | 0.516850 | 0.472437 | 0.044412 | 125 | 23 | 0.837838 | 0.371622 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 148 | 0.594 | 0.516850 | 0.472437 | 0.044412 | 125 | 23 | 0.837838 | 0.371622 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 249 | 1.000 | 0.629771 | 0.603497 | 0.026275 | 161 | 88 | 0.843373 | 0.566265 |

## Active Smoke/Inferno Intervals

- `10.0s` - `78.0s`, rows `137`
- `80.0s` - `85.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.5`, LSTM `0.4256`, XGBoost `0.1724`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.4227`, XGBoost `0.1720`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.4152`, XGBoost `0.1739`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.4134`, XGBoost `0.1724`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.4054`, XGBoost `0.1724`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.3839`, XGBoost `0.1691`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.3947`, XGBoost `0.2494`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5203`, XGBoost `0.3935`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5196`, XGBoost `0.3930`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.2853`, XGBoost `0.1665`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
