# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `3`
- rows: `161`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 161 | 1.000 | 0.044947 | 0.068170 | -0.023222 | 159 | 2 | 1.000000 | 1.000000 |
| active/recent utility | 161 | 1.000 | 0.044947 | 0.068170 | -0.023222 | 159 | 2 | 1.000000 | 1.000000 |
| strong utility action | 81 | 0.503 | 0.047793 | 0.078483 | -0.030690 | 80 | 1 | 1.000000 | 1.000000 |
| utility damage | 16 | 0.099 | 0.202509 | 0.268406 | -0.065897 | 15 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 81 | 0.503 | 0.047793 | 0.078483 | -0.030690 | 80 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 161 | 1.000 | 0.044947 | 0.068170 | -0.023222 | 159 | 2 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `48.5s`, rows `81`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.0490`, XGBoost `0.1659`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0554`, XGBoost `0.1716`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.0593`, XGBoost `0.1716`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0602`, XGBoost `0.1719`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.2312`, XGBoost `0.3425`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0544`, XGBoost `0.1654`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0591`, XGBoost `0.1698`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0606`, XGBoost `0.1705`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0597`, XGBoost `0.1678`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.0630`, XGBoost `0.1703`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
