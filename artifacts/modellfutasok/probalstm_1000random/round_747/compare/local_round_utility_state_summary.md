# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `3`
- rows: `223`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 223 | 1.000 | 0.463751 | 0.475224 | -0.011472 | 69 | 154 | 0.587444 | 0.825112 |
| active/recent utility | 223 | 1.000 | 0.463751 | 0.475224 | -0.011472 | 69 | 154 | 0.587444 | 0.825112 |
| strong utility action | 199 | 0.892 | 0.467935 | 0.480782 | -0.012847 | 64 | 135 | 0.587940 | 0.804020 |
| utility damage | 11 | 0.049 | 0.269680 | 0.490810 | -0.221130 | 11 | 0 | 1.000000 | 0.545455 |
| active smoke/inferno | 199 | 0.892 | 0.467935 | 0.480782 | -0.012847 | 64 | 135 | 0.587940 | 0.804020 |
| recent utility last 5s | 20 | 0.090 | 0.474295 | 0.521744 | -0.047449 | 8 | 12 | 0.700000 | 0.700000 |
| flash effect present | 223 | 1.000 | 0.463751 | 0.475224 | -0.011472 | 69 | 154 | 0.587444 | 0.825112 |

## Active Smoke/Inferno Intervals

- `9.5s` - `108.5s`, rows `199`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `105.5`, LSTM `0.4381`, XGBoost `0.7110`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.3280`, XGBoost `0.5939`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.1217`, XGBoost `0.3842`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.1260`, XGBoost `0.3790`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `108.5`, LSTM `0.1535`, XGBoost `0.4051`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.1298`, XGBoost `0.3790`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.1363`, XGBoost `0.3842`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.3595`, XGBoost `0.6057`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.1301`, XGBoost `0.3727`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `5.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.2984`, XGBoost `0.5399`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
