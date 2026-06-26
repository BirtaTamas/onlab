# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `16`
- rows: `205`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.514215 | 0.530912 | -0.016697 | 135 | 70 | 0.160976 | 0.263415 |
| active/recent utility | 205 | 1.000 | 0.514215 | 0.530912 | -0.016697 | 135 | 70 | 0.160976 | 0.263415 |
| strong utility action | 182 | 0.888 | 0.512931 | 0.532891 | -0.019960 | 126 | 56 | 0.159341 | 0.274725 |
| utility damage | 10 | 0.049 | 0.604263 | 0.597501 | 0.006762 | 4 | 6 | 0.000000 | 0.000000 |
| active smoke/inferno | 182 | 0.888 | 0.512931 | 0.532891 | -0.019960 | 126 | 56 | 0.159341 | 0.274725 |
| recent utility last 5s | 10 | 0.049 | 0.400588 | 0.320427 | 0.080161 | 2 | 8 | 0.400000 | 1.000000 |
| flash effect present | 205 | 1.000 | 0.514215 | 0.530912 | -0.016697 | 135 | 70 | 0.160976 | 0.263415 |

## Active Smoke/Inferno Intervals

- `6.5s` - `68.0s`, rows `124`
- `71.5s` - `100.0s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.0`, LSTM `0.1096`, XGBoost `0.3819`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.1337`, XGBoost `0.3992`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.1273`, XGBoost `0.3819`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.1417`, XGBoost `0.3956`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.1280`, XGBoost `0.3819`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.1343`, XGBoost `0.3879`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.0994`, XGBoost `0.3319`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.1503`, XGBoost `0.3804`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.0948`, XGBoost `0.3106`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.1866`, XGBoost `0.3992`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
