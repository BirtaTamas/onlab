# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `14`
- rows: `262`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 262 | 1.000 | 0.131474 | 0.187322 | -0.055847 | 261 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 262 | 1.000 | 0.131474 | 0.187322 | -0.055847 | 261 | 1 | 1.000000 | 1.000000 |
| strong utility action | 195 | 0.744 | 0.138158 | 0.195590 | -0.057432 | 194 | 1 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 195 | 0.744 | 0.138158 | 0.195590 | -0.057432 | 194 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 262 | 1.000 | 0.131474 | 0.187322 | -0.055847 | 261 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `30.5s`, rows `49`
- `39.5s` - `70.0s`, rows `62`
- `72.5s` - `114.0s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.0`, LSTM `0.1623`, XGBoost `0.3274`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1410`, XGBoost `0.2993`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1827`, XGBoost `0.3279`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1901`, XGBoost `0.3279`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.1242`, XGBoost `0.2613`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.1911`, XGBoost `0.3279`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1864`, XGBoost `0.3225`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.1868`, XGBoost `0.3225`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.1268`, XGBoost `0.2613`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.1294`, XGBoost `0.2621`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
