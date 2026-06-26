# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `12`
- rows: `177`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.174054 | 0.240163 | -0.066109 | 177 | 0 | 1.000000 | 0.711864 |
| active/recent utility | 177 | 1.000 | 0.174054 | 0.240163 | -0.066109 | 177 | 0 | 1.000000 | 0.711864 |
| strong utility action | 131 | 0.740 | 0.173832 | 0.235250 | -0.061418 | 131 | 0 | 1.000000 | 0.679389 |
| utility damage | 10 | 0.056 | 0.003895 | 0.010300 | -0.006405 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 131 | 0.740 | 0.173832 | 0.235250 | -0.061418 | 131 | 0 | 1.000000 | 0.679389 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 177 | 1.000 | 0.174054 | 0.240163 | -0.066109 | 177 | 0 | 1.000000 | 0.711864 |

## Active Smoke/Inferno Intervals

- `9.5s` - `53.5s`, rows `89`
- `67.5s` - `88.0s`, rows `42`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.1340`, XGBoost `0.3174`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.1426`, XGBoost `0.3179`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.1435`, XGBoost `0.3179`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.1455`, XGBoost `0.3179`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.1463`, XGBoost `0.3179`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.1542`, XGBoost `0.3179`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.1305`, XGBoost `0.2934`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1616`, XGBoost `0.3179`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.3749`, XGBoost `0.5052`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.1916`, XGBoost `0.3174`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
