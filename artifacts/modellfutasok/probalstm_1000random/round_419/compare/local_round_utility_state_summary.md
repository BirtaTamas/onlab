# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-natus-vincere-bo3-FVT9m_t7tlOrOuiYTIheUW/the-mongolz-vs-natus-vincere-m2-inferno.csv`
- round_num: `12`
- rows: `238`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 238 | 1.000 | 0.730081 | 0.715802 | 0.014280 | 119 | 119 | 1.000000 | 0.974790 |
| active/recent utility | 238 | 1.000 | 0.730081 | 0.715802 | 0.014280 | 119 | 119 | 1.000000 | 0.974790 |
| strong utility action | 188 | 0.790 | 0.714446 | 0.707939 | 0.006507 | 87 | 101 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.084 | 0.603933 | 0.623660 | -0.019727 | 4 | 16 | 1.000000 | 1.000000 |
| active smoke/inferno | 180 | 0.756 | 0.722385 | 0.715056 | 0.007329 | 86 | 94 | 1.000000 | 1.000000 |
| recent utility last 5s | 8 | 0.034 | 0.535835 | 0.547807 | -0.011972 | 1 | 7 | 1.000000 | 1.000000 |
| flash effect present | 238 | 1.000 | 0.730081 | 0.715802 | 0.014280 | 119 | 119 | 1.000000 | 0.974790 |

## Active Smoke/Inferno Intervals

- `10.0s` - `40.5s`, rows `60`
- `41.5s` - `101.5s`, rows `120`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.7286`, XGBoost `0.5962`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.7297`, XGBoost `0.6242`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.7205`, XGBoost `0.6278`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.7682`, XGBoost `0.6755`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.7667`, XGBoost `0.6755`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.7658`, XGBoost `0.6755`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.7635`, XGBoost `0.6755`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.7612`, XGBoost `0.6755`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.7590`, XGBoost `0.6753`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6374`, XGBoost `0.5554`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
