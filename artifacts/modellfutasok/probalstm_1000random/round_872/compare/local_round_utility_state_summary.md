# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `12`
- rows: `243`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 243 | 1.000 | 0.371710 | 0.432376 | -0.060667 | 243 | 0 | 0.670782 | 0.279835 |
| active/recent utility | 243 | 1.000 | 0.371710 | 0.432376 | -0.060667 | 243 | 0 | 0.670782 | 0.279835 |
| strong utility action | 174 | 0.716 | 0.448793 | 0.517272 | -0.068479 | 174 | 0 | 0.540230 | 0.137931 |
| utility damage | 10 | 0.041 | 0.535463 | 0.611399 | -0.075936 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 166 | 0.683 | 0.446275 | 0.512892 | -0.066617 | 166 | 0 | 0.542169 | 0.144578 |
| recent utility last 5s | 10 | 0.041 | 0.500467 | 0.608100 | -0.107633 | 10 | 0 | 0.500000 | 0.000000 |
| flash effect present | 243 | 1.000 | 0.371710 | 0.432376 | -0.060667 | 243 | 0 | 0.670782 | 0.279835 |

## Active Smoke/Inferno Intervals

- `3.0s` - `39.5s`, rows `74`
- `53.5s` - `99.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `91.0`, LSTM `0.1662`, XGBoost `0.3153`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.1617`, XGBoost `0.2999`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.1649`, XGBoost `0.2999`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.2548`, XGBoost `0.3798`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.4915`, XGBoost `0.6117`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.4882`, XGBoost `0.6081`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `39.0`, LSTM `0.4909`, XGBoost `0.6079`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `32.5`, LSTM `0.4954`, XGBoost `0.6112`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.4956`, XGBoost `0.6113`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.4980`, XGBoost `0.6117`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
