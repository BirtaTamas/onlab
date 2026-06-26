# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `1`
- rows: `203`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 203 | 1.000 | 0.658043 | 0.787561 | -0.129518 | 21 | 182 | 1.000000 | 0.985222 |
| active/recent utility | 203 | 1.000 | 0.658043 | 0.787561 | -0.129518 | 21 | 182 | 1.000000 | 0.985222 |
| strong utility action | 69 | 0.340 | 0.567470 | 0.725113 | -0.157643 | 4 | 65 | 1.000000 | 0.971014 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 69 | 0.340 | 0.567470 | 0.725113 | -0.157643 | 4 | 65 | 1.000000 | 0.971014 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 203 | 1.000 | 0.658043 | 0.787561 | -0.129518 | 21 | 182 | 1.000000 | 0.985222 |

## Active Smoke/Inferno Intervals

- `8.5s` - `42.5s`, rows `69`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.5371`, XGBoost `0.7750`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5380`, XGBoost `0.7750`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5469`, XGBoost `0.7790`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5386`, XGBoost `0.7697`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5393`, XGBoost `0.7697`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5473`, XGBoost `0.7764`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5423`, XGBoost `0.7697`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5427`, XGBoost `0.7697`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5435`, XGBoost `0.7697`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5518`, XGBoost `0.7773`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
