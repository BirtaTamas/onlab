# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `3`
- rows: `263`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 263 | 1.000 | 0.044583 | 0.090740 | -0.046157 | 259 | 4 | 0.977186 | 0.946768 |
| active/recent utility | 263 | 1.000 | 0.044583 | 0.090740 | -0.046157 | 259 | 4 | 0.977186 | 0.946768 |
| strong utility action | 149 | 0.567 | 0.019220 | 0.058162 | -0.038942 | 149 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 149 | 0.567 | 0.019220 | 0.058162 | -0.038942 | 149 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 263 | 1.000 | 0.044583 | 0.090740 | -0.046157 | 259 | 4 | 0.977186 | 0.946768 |

## Active Smoke/Inferno Intervals

- `8.5s` - `37.0s`, rows `58`
- `56.5s` - `101.5s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `95.0`, LSTM `0.0264`, XGBoost `0.1168`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.0362`, XGBoost `0.1254`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.0230`, XGBoost `0.1111`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.0182`, XGBoost `0.1062`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.0396`, XGBoost `0.1259`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.0332`, XGBoost `0.1174`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.0275`, XGBoost `0.1112`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.0294`, XGBoost `0.1123`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.0284`, XGBoost `0.1113`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.0210`, XGBoost `0.1036`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
