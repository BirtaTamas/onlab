# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `17`
- rows: `281`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 281 | 1.000 | 0.135059 | 0.131463 | 0.003596 | 205 | 76 | 0.911032 | 0.846975 |
| active/recent utility | 281 | 1.000 | 0.135059 | 0.131463 | 0.003596 | 205 | 76 | 0.911032 | 0.846975 |
| strong utility action | 201 | 0.715 | 0.121924 | 0.120668 | 0.001256 | 140 | 61 | 0.895522 | 0.885572 |
| utility damage | 37 | 0.132 | 0.105360 | 0.133135 | -0.027775 | 18 | 19 | 1.000000 | 1.000000 |
| active smoke/inferno | 201 | 0.715 | 0.121924 | 0.120668 | 0.001256 | 140 | 61 | 0.895522 | 0.885572 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 281 | 1.000 | 0.135059 | 0.131463 | 0.003596 | 205 | 76 | 0.911032 | 0.846975 |

## Active Smoke/Inferno Intervals

- `10.0s` - `36.5s`, rows `54`
- `37.5s` - `42.5s`, rows `11`
- `49.0s` - `93.5s`, rows `90`
- `99.0s` - `121.5s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.5`, LSTM `0.4821`, XGBoost `0.3226`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3040`, XGBoost `0.1539`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.3025`, XGBoost `0.1557`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.2921`, XGBoost `0.1507`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.2721`, XGBoost `0.1511`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.2870`, XGBoost `0.1661`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.2711`, XGBoost `0.1511`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.2855`, XGBoost `0.1661`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0370`, XGBoost `0.1547`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `167.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.2707`, XGBoost `0.1532`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
