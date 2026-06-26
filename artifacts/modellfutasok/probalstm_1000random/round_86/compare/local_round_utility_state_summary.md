# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `7`
- rows: `112`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 112 | 1.000 | 0.894327 | 0.888826 | 0.005502 | 31 | 81 | 1.000000 | 1.000000 |
| active/recent utility | 112 | 1.000 | 0.894327 | 0.888826 | 0.005502 | 31 | 81 | 1.000000 | 1.000000 |
| strong utility action | 100 | 0.893 | 0.925902 | 0.928017 | -0.002115 | 19 | 81 | 1.000000 | 1.000000 |
| utility damage | 32 | 0.286 | 0.899041 | 0.893225 | 0.005816 | 8 | 24 | 1.000000 | 1.000000 |
| active smoke/inferno | 100 | 0.893 | 0.925902 | 0.928017 | -0.002115 | 19 | 81 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 112 | 1.000 | 0.894327 | 0.888826 | 0.005502 | 31 | 81 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `55.5s`, rows `100`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.5`, LSTM `0.7121`, XGBoost `0.5713`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.7166`, XGBoost `0.5790`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.7039`, XGBoost `0.5790`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.6939`, XGBoost `0.5713`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.6896`, XGBoost `0.5692`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.6397`, XGBoost `0.5642`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.8222`, XGBoost `0.7524`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.8153`, XGBoost `0.7526`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.8091`, XGBoost `0.7524`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.8026`, XGBoost `0.7524`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `20.0`, recent_utility `0`
