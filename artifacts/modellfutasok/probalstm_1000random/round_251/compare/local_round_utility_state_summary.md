# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `3`
- rows: `153`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.768729 | 0.791360 | -0.022631 | 35 | 118 | 0.869281 | 0.790850 |
| active/recent utility | 153 | 1.000 | 0.768729 | 0.791360 | -0.022631 | 35 | 118 | 0.869281 | 0.790850 |
| strong utility action | 140 | 0.915 | 0.794794 | 0.820591 | -0.025798 | 23 | 117 | 0.935714 | 0.864286 |
| utility damage | 17 | 0.111 | 0.640551 | 0.667349 | -0.026798 | 5 | 12 | 0.764706 | 0.764706 |
| active smoke/inferno | 140 | 0.915 | 0.794794 | 0.820591 | -0.025798 | 23 | 117 | 0.935714 | 0.864286 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.768729 | 0.791360 | -0.022631 | 35 | 118 | 0.869281 | 0.790850 |

## Active Smoke/Inferno Intervals

- `6.5s` - `76.0s`, rows `140`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `16.5`, LSTM `0.6249`, XGBoost `0.7337`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.6841`, XGBoost `0.7605`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.5348`, XGBoost `0.4599`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.5313`, XGBoost `0.4601`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.5290`, XGBoost `0.4601`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.5286`, XGBoost `0.4607`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.6460`, XGBoost `0.7136`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `9.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.9201`, XGBoost `0.9863`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.7009`, XGBoost `0.7670`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.6643`, XGBoost `0.7283`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
