# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `10`
- rows: `200`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.804050 | 0.709874 | 0.094176 | 169 | 31 | 0.985000 | 0.980000 |
| active/recent utility | 200 | 1.000 | 0.804050 | 0.709874 | 0.094176 | 169 | 31 | 0.985000 | 0.980000 |
| strong utility action | 167 | 0.835 | 0.795653 | 0.696725 | 0.098928 | 150 | 17 | 0.982036 | 0.976048 |
| utility damage | 11 | 0.055 | 0.572005 | 0.477825 | 0.094180 | 10 | 1 | 0.727273 | 0.727273 |
| active smoke/inferno | 163 | 0.815 | 0.800377 | 0.700963 | 0.099414 | 146 | 17 | 0.981595 | 0.975460 |
| recent utility last 5s | 21 | 0.105 | 0.715960 | 0.603934 | 0.112026 | 21 | 0 | 1.000000 | 1.000000 |
| flash effect present | 200 | 1.000 | 0.804050 | 0.709874 | 0.094176 | 169 | 31 | 0.985000 | 0.980000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `36.0s`, rows `67`
- `40.0s` - `46.5s`, rows `14`
- `49.0s` - `89.5s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.7855`, XGBoost `0.5341`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.7820`, XGBoost `0.5341`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7723`, XGBoost `0.5341`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.7828`, XGBoost `0.5478`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.7674`, XGBoost `0.5341`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.7809`, XGBoost `0.5478`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.7743`, XGBoost `0.5478`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7477`, XGBoost `0.5341`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7471`, XGBoost `0.5341`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.7451`, XGBoost `0.5341`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
