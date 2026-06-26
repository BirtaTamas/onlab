# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `18`
- rows: `146`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.016634 | 0.070026 | -0.053391 | 105 | 41 | 1.000000 | 1.000000 |
| active/recent utility | 146 | 1.000 | 0.016634 | 0.070026 | -0.053391 | 105 | 41 | 1.000000 | 1.000000 |
| strong utility action | 113 | 0.774 | 0.018579 | 0.081267 | -0.062689 | 99 | 14 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 103 | 0.705 | 0.017146 | 0.073639 | -0.056493 | 89 | 14 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.068 | 0.033331 | 0.159838 | -0.126507 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 146 | 1.000 | 0.016634 | 0.070026 | -0.053391 | 105 | 41 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `59.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.0201`, XGBoost `0.1756`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0197`, XGBoost `0.1739`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0235`, XGBoost `0.1772`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0245`, XGBoost `0.1772`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0260`, XGBoost `0.1772`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0270`, XGBoost `0.1721`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0271`, XGBoost `0.1719`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `5.5`, LSTM `0.0332`, XGBoost `0.1779`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `17.5`, LSTM `0.0309`, XGBoost `0.1742`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0340`, XGBoost `0.1767`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
