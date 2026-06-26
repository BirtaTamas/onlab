# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `4`
- rows: `187`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.245636 | 0.245618 | 0.000018 | 140 | 47 | 0.598930 | 0.743316 |
| active/recent utility | 187 | 1.000 | 0.245636 | 0.245618 | 0.000018 | 140 | 47 | 0.598930 | 0.743316 |
| strong utility action | 124 | 0.663 | 0.285392 | 0.287754 | -0.002362 | 97 | 27 | 0.540323 | 0.612903 |
| utility damage | 25 | 0.134 | 0.553075 | 0.558869 | -0.005793 | 15 | 10 | 0.000000 | 0.200000 |
| active smoke/inferno | 124 | 0.663 | 0.285392 | 0.287754 | -0.002362 | 97 | 27 | 0.540323 | 0.612903 |
| recent utility last 5s | 10 | 0.053 | 0.007450 | 0.045868 | -0.038418 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 187 | 1.000 | 0.245636 | 0.245618 | 0.000018 | 140 | 47 | 0.598930 | 0.743316 |

## Active Smoke/Inferno Intervals

- `10.0s` - `71.5s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.5179`, XGBoost `0.3225`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.3253`, XGBoost `0.1730`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.4781`, XGBoost `0.3290`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.4984`, XGBoost `0.3713`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.2082`, XGBoost `0.0858`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5105`, XGBoost `0.3914`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5085`, XGBoost `0.3914`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.4478`, XGBoost `0.3330`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.1675`, XGBoost `0.0854`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.1480`, XGBoost `0.0849`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
