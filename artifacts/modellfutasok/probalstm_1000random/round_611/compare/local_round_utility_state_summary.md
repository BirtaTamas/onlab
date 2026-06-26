# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `10`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.487978 | 0.632095 | -0.144117 | 0 | 135 | 0.385185 | 0.585185 |
| active/recent utility | 135 | 1.000 | 0.487978 | 0.632095 | -0.144117 | 0 | 135 | 0.385185 | 0.585185 |
| strong utility action | 121 | 0.896 | 0.497409 | 0.648815 | -0.151406 | 0 | 121 | 0.429752 | 0.652893 |
| utility damage | 20 | 0.148 | 0.485776 | 0.680250 | -0.194474 | 0 | 20 | 0.350000 | 1.000000 |
| active smoke/inferno | 121 | 0.896 | 0.497409 | 0.648815 | -0.151406 | 0 | 121 | 0.429752 | 0.652893 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 135 | 1.000 | 0.487978 | 0.632095 | -0.144117 | 0 | 135 | 0.385185 | 0.585185 |

## Active Smoke/Inferno Intervals

- `7.0s` - `67.0s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.4877`, XGBoost `0.7539`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.4962`, XGBoost `0.7603`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4991`, XGBoost `0.7603`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.4922`, XGBoost `0.7514`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5952`, XGBoost `0.8479`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `19.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.1993`, XGBoost `0.4516`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.6033`, XGBoost `0.8468`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `19.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.1950`, XGBoost `0.4376`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1991`, XGBoost `0.4367`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2018`, XGBoost `0.4376`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
