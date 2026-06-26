# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-lynn-vision-bo3-KVSQ5iZB0TjTG70slfdqOB/furia-vs-lynn-vision-m2-overpass.csv`
- round_num: `12`
- rows: `175`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 175 | 1.000 | 0.261876 | 0.290814 | -0.028939 | 158 | 17 | 0.605714 | 0.571429 |
| active/recent utility | 175 | 1.000 | 0.261876 | 0.290814 | -0.028939 | 158 | 17 | 0.605714 | 0.571429 |
| strong utility action | 136 | 0.777 | 0.240421 | 0.272629 | -0.032208 | 128 | 8 | 0.632353 | 0.588235 |
| utility damage | 30 | 0.171 | 0.300864 | 0.321143 | -0.020280 | 29 | 1 | 0.433333 | 0.433333 |
| active smoke/inferno | 132 | 0.754 | 0.247558 | 0.280406 | -0.032847 | 124 | 8 | 0.621212 | 0.575758 |
| recent utility last 5s | 10 | 0.057 | 0.005163 | 0.016862 | -0.011698 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 175 | 1.000 | 0.261876 | 0.290814 | -0.028939 | 158 | 17 | 0.605714 | 0.571429 |

## Active Smoke/Inferno Intervals

- `8.5s` - `36.0s`, rows `56`
- `44.5s` - `76.5s`, rows `65`
- `78.0s` - `83.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.5`, LSTM `0.2867`, XGBoost `0.4450`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `58.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.2954`, XGBoost `0.4524`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `50.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.3400`, XGBoost `0.4625`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0313`, XGBoost `0.1408`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0313`, XGBoost `0.1408`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0315`, XGBoost `0.1408`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0329`, XGBoost `0.1403`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.0359`, XGBoost `0.1408`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.0424`, XGBoost `0.1459`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `57.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.3725`, XGBoost `0.4756`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `3.0`, recent_utility `0`
