# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `13`
- rows: `174`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 174 | 1.000 | 0.619854 | 0.671845 | -0.051991 | 22 | 152 | 0.568966 | 0.948276 |
| active/recent utility | 174 | 1.000 | 0.619854 | 0.671845 | -0.051991 | 22 | 152 | 0.568966 | 0.948276 |
| strong utility action | 50 | 0.287 | 0.715673 | 0.845553 | -0.129880 | 0 | 50 | 0.860000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 50 | 0.287 | 0.715673 | 0.845553 | -0.129880 | 0 | 50 | 0.860000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 174 | 1.000 | 0.619854 | 0.671845 | -0.051991 | 22 | 152 | 0.568966 | 0.948276 |

## Active Smoke/Inferno Intervals

- `49.5s` - `74.0s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.5779`, XGBoost `0.8421`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5840`, XGBoost `0.8427`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5837`, XGBoost `0.8415`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.6274`, XGBoost `0.8735`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.5961`, XGBoost `0.8415`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.6075`, XGBoost `0.8463`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.6051`, XGBoost `0.8431`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.6393`, XGBoost `0.8737`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.6153`, XGBoost `0.8471`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.6162`, XGBoost `0.8471`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
