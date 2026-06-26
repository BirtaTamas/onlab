# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `1`
- rows: `126`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 126 | 1.000 | 0.318065 | 0.351772 | -0.033707 | 116 | 10 | 0.603175 | 0.642857 |
| active/recent utility | 126 | 1.000 | 0.318065 | 0.351772 | -0.033707 | 116 | 10 | 0.603175 | 0.642857 |
| strong utility action | 44 | 0.349 | 0.265666 | 0.348924 | -0.083257 | 41 | 3 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.349 | 0.265666 | 0.348924 | -0.083257 | 41 | 3 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 126 | 1.000 | 0.318065 | 0.351772 | -0.033707 | 116 | 10 | 0.603175 | 0.642857 |

## Active Smoke/Inferno Intervals

- `27.5s` - `49.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.0`, LSTM `0.2152`, XGBoost `0.4273`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.2494`, XGBoost `0.4273`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1717`, XGBoost `0.3147`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1860`, XGBoost `0.3269`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1912`, XGBoost `0.3269`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.1919`, XGBoost `0.3269`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.1920`, XGBoost `0.3256`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1964`, XGBoost `0.3269`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.1993`, XGBoost `0.3256`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.3065`, XGBoost `0.4314`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
