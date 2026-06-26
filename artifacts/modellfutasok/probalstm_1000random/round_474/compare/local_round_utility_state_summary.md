# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `17`
- rows: `310`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 310 | 1.000 | 0.479689 | 0.441555 | 0.038134 | 81 | 229 | 0.316129 | 0.393548 |
| active/recent utility | 310 | 1.000 | 0.479689 | 0.441555 | 0.038134 | 81 | 229 | 0.316129 | 0.393548 |
| strong utility action | 224 | 0.723 | 0.500298 | 0.483727 | 0.016571 | 78 | 146 | 0.285714 | 0.343750 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 224 | 0.723 | 0.500298 | 0.483727 | 0.016571 | 78 | 146 | 0.285714 | 0.343750 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 310 | 1.000 | 0.479689 | 0.441555 | 0.038134 | 81 | 229 | 0.316129 | 0.393548 |

## Active Smoke/Inferno Intervals

- `6.5s` - `40.5s`, rows `69`
- `42.0s` - `70.0s`, rows `57`
- `89.0s` - `115.5s`, rows `54`
- `120.0s` - `141.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `109.5`, LSTM `0.3761`, XGBoost `0.2200`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.6023`, XGBoost `0.4642`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.6066`, XGBoost `0.4749`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.6168`, XGBoost `0.4923`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.8157`, XGBoost `0.6919`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.6136`, XGBoost `0.4923`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.6063`, XGBoost `0.4863`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.5866`, XGBoost `0.4672`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6338`, XGBoost `0.5190`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.5799`, XGBoost `0.4672`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
