# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `9`
- rows: `251`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 251 | 1.000 | 0.203782 | 0.337909 | -0.134127 | 251 | 0 | 0.940239 | 0.788845 |
| active/recent utility | 251 | 1.000 | 0.203782 | 0.337909 | -0.134127 | 251 | 0 | 0.940239 | 0.788845 |
| strong utility action | 202 | 0.805 | 0.184131 | 0.323738 | -0.139607 | 202 | 0 | 0.990099 | 0.836634 |
| utility damage | 31 | 0.124 | 0.218253 | 0.350204 | -0.131951 | 31 | 0 | 1.000000 | 0.741935 |
| active smoke/inferno | 197 | 0.785 | 0.186718 | 0.325861 | -0.139143 | 197 | 0 | 0.989848 | 0.832487 |
| recent utility last 5s | 11 | 0.044 | 0.076303 | 0.351366 | -0.275063 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 251 | 1.000 | 0.203782 | 0.337909 | -0.134127 | 251 | 0 | 0.940239 | 0.788845 |

## Active Smoke/Inferno Intervals

- `3.5s` - `37.5s`, rows `69`
- `41.0s` - `99.0s`, rows `117`
- `106.0s` - `111.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.5`, LSTM `0.0383`, XGBoost `0.3386`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0419`, XGBoost `0.3389`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0435`, XGBoost `0.3389`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0446`, XGBoost `0.3388`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0466`, XGBoost `0.3386`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0475`, XGBoost `0.3386`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.0906`, XGBoost `0.3797`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0531`, XGBoost `0.3400`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.0939`, XGBoost `0.3805`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0531`, XGBoost `0.3388`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
