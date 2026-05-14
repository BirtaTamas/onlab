# Local Round Utility State Analysis

- csv_path: `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv`
- round_num: `2`
- rows: `250`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 250 | 1.000 | 0.499396 | 0.568420 | -0.069024 | 46 | 204 | 0.376000 | 0.644000 |
| active/recent utility | 250 | 1.000 | 0.499396 | 0.568420 | -0.069024 | 46 | 204 | 0.376000 | 0.644000 |
| strong utility action | 180 | 0.720 | 0.434651 | 0.507103 | -0.072452 | 38 | 142 | 0.311111 | 0.555556 |
| utility damage | 10 | 0.040 | 0.491350 | 0.510059 | -0.018708 | 0 | 10 | 0.200000 | 1.000000 |
| active smoke/inferno | 180 | 0.720 | 0.434651 | 0.507103 | -0.072452 | 38 | 142 | 0.311111 | 0.555556 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 250 | 1.000 | 0.499396 | 0.568420 | -0.069024 | 46 | 204 | 0.376000 | 0.644000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `77.5s`, rows `136`
- `89.5s` - `111.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.5`, LSTM `0.4711`, XGBoost `0.8375`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.4919`, XGBoost `0.8398`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.4953`, XGBoost `0.8379`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.4957`, XGBoost `0.8235`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.5632`, XGBoost `0.8383`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.5677`, XGBoost `0.8383`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.3538`, XGBoost `0.6146`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.2987`, XGBoost `0.5464`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.3085`, XGBoost `0.5466`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.3313`, XGBoost `0.5642`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
