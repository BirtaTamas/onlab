# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `9`
- rows: `160`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 160 | 1.000 | 0.363959 | 0.464749 | -0.100790 | 158 | 2 | 0.631250 | 0.281250 |
| active/recent utility | 160 | 1.000 | 0.363959 | 0.464749 | -0.100790 | 158 | 2 | 0.631250 | 0.281250 |
| strong utility action | 157 | 0.981 | 0.362137 | 0.463429 | -0.101292 | 155 | 2 | 0.624204 | 0.286624 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 148 | 0.925 | 0.357039 | 0.459204 | -0.102166 | 146 | 2 | 0.601351 | 0.304054 |
| recent utility last 5s | 20 | 0.125 | 0.257118 | 0.359313 | -0.102195 | 20 | 0 | 1.000000 | 0.500000 |
| flash effect present | 160 | 1.000 | 0.363959 | 0.464749 | -0.100790 | 158 | 2 | 0.631250 | 0.281250 |

## Active Smoke/Inferno Intervals

- `6.0s` - `79.5s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.5`, LSTM `0.4934`, XGBoost `0.7635`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.2737`, XGBoost `0.5323`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5119`, XGBoost `0.7659`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5148`, XGBoost `0.7659`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5163`, XGBoost `0.7659`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5175`, XGBoost `0.7659`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5186`, XGBoost `0.7659`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5200`, XGBoost `0.7659`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2866`, XGBoost `0.5323`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5252`, XGBoost `0.7659`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
