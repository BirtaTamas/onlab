# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `4`
- rows: `178`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.186566 | 0.245263 | -0.058697 | 161 | 17 | 0.977528 | 0.870787 |
| active/recent utility | 178 | 1.000 | 0.186566 | 0.245263 | -0.058697 | 161 | 17 | 0.977528 | 0.870787 |
| strong utility action | 129 | 0.725 | 0.206912 | 0.279820 | -0.072908 | 112 | 17 | 0.968992 | 0.930233 |
| utility damage | 10 | 0.056 | 0.264369 | 0.275236 | -0.010867 | 5 | 5 | 0.700000 | 1.000000 |
| active smoke/inferno | 129 | 0.725 | 0.206912 | 0.279820 | -0.072908 | 112 | 17 | 0.968992 | 0.930233 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 178 | 1.000 | 0.186566 | 0.245263 | -0.058697 | 161 | 17 | 0.977528 | 0.870787 |

## Active Smoke/Inferno Intervals

- `7.0s` - `71.0s`, rows `129`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.0`, LSTM `0.1847`, XGBoost `0.4676`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.2035`, XGBoost `0.4664`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.2105`, XGBoost `0.4664`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.0466`, XGBoost `0.2756`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0456`, XGBoost `0.2731`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0492`, XGBoost `0.2756`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.0476`, XGBoost `0.2734`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0391`, XGBoost `0.2646`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.0606`, XGBoost `0.2827`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0350`, XGBoost `0.2524`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
