# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-natus-vincere-bo3-FVT9m_t7tlOrOuiYTIheUW/the-mongolz-vs-natus-vincere-m2-inferno.csv`
- round_num: `10`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.849429 | 0.838300 | 0.011130 | 105 | 107 | 1.000000 | 0.952830 |
| active/recent utility | 212 | 1.000 | 0.849429 | 0.838300 | 0.011130 | 105 | 107 | 1.000000 | 0.952830 |
| strong utility action | 181 | 0.854 | 0.870579 | 0.867839 | 0.002740 | 84 | 97 | 1.000000 | 0.983425 |
| utility damage | 31 | 0.146 | 0.882481 | 0.875364 | 0.007118 | 19 | 12 | 1.000000 | 1.000000 |
| active smoke/inferno | 181 | 0.854 | 0.870579 | 0.867839 | 0.002740 | 84 | 97 | 1.000000 | 0.983425 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 212 | 1.000 | 0.849429 | 0.838300 | 0.011130 | 105 | 107 | 1.000000 | 0.952830 |

## Active Smoke/Inferno Intervals

- `10.0s` - `94.5s`, rows `170`
- `96.5s` - `101.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.0`, LSTM `0.6612`, XGBoost `0.5018`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6545`, XGBoost `0.4957`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6513`, XGBoost `0.4950`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6543`, XGBoost `0.4986`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6563`, XGBoost `0.5017`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.7120`, XGBoost `0.6057`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.9176`, XGBoost `0.8788`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `9.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.8501`, XGBoost `0.8878`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.8525`, XGBoost `0.8893`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.8411`, XGBoost `0.8775`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
