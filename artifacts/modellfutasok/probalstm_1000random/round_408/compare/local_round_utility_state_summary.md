# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `15`
- rows: `127`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 127 | 1.000 | 0.123701 | 0.220737 | -0.097036 | 127 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 127 | 1.000 | 0.123701 | 0.220737 | -0.097036 | 127 | 0 | 1.000000 | 1.000000 |
| strong utility action | 106 | 0.835 | 0.081589 | 0.182424 | -0.100835 | 106 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.079 | 0.272711 | 0.411668 | -0.138957 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 106 | 0.835 | 0.081589 | 0.182424 | -0.100835 | 106 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 17 | 0.134 | 0.002016 | 0.024997 | -0.022981 | 17 | 0 | 1.000000 | 1.000000 |
| flash effect present | 127 | 1.000 | 0.123701 | 0.220737 | -0.097036 | 127 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `63.0s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.5`, LSTM `0.0428`, XGBoost `0.2265`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0437`, XGBoost `0.2260`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.2118`, XGBoost `0.3939`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `30.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0431`, XGBoost `0.2252`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0456`, XGBoost `0.2265`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0457`, XGBoost `0.2265`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.0463`, XGBoost `0.2258`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0471`, XGBoost `0.2260`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.0477`, XGBoost `0.2265`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0449`, XGBoost `0.2236`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
